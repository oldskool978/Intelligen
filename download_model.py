# download_model.py
import os
import sys
import gc
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CACHE_DIR = ROOT_DIR / ".hf_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

try:
    from refinery import refineTensor, clear_refinery_cache
    HAS_REFINERY = True
except ImportError:
    HAS_REFINERY = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

PRECISION_CHOICES = ["native", "bf16", "fp16", "e4m3fn", "e5m2"]
QUANT_ENGINES = ["clean_rtn", "dto_ca"]

DTYPE_MAP = {
    "native": None,
    "none": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "e4m3fn": getattr(torch, "float8_e4m3fn", None),
    "e5m2": getattr(torch, "float8_e5m2", None),
}

DTYPE_LOOKUP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "e4m3fn": getattr(torch, "float8_e4m3fn", None),
    "e5m2": getattr(torch, "float8_e5m2", None),
    "e8m0fnu": getattr(torch, "float8_e8m0fnu", None),
}

FP8_DTYPES = [v for k, v in DTYPE_LOOKUP.items() if "e" in k and v is not None]

IGNORE_DOWNLOAD_PATTERNS = [
    "qwen_7B/*",
    "qwen_7B*",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.msgpack",
    "*.h5"
]

PROTECTED_NAMESPACES = [
    "vocoder",
    "flow_vae",
    "rvq_depth_decoder",
    "rvq",
    "audio_vae",
    "visual_vae",
    "condition_encoder",
    "text_encoder"
]

PROTECTED_KEY_PATTERNS = [
    "norm",
    "bias",
    "embed",
    "scale",
    "inv_freq",
    "pos_embed",
    "alpha",
    "beta",
    "weight_g",
    "snake",
    "filter",
    "kaiser",
    "sinc"
]

DEAD_KEY_PATTERNS = [
    "rotary_emb.inv_freq",
    "loss",
    "training",
    "optimizer",
    "total_ops",
    "total_params"
]

def is_dead_key(key: str) -> bool:
    key_lower = key.lower()
    return any(pattern in key_lower for pattern in DEAD_KEY_PATTERNS)

def is_quantizable(key: str, file_path: Path, tensor: torch.Tensor) -> bool:
    if not tensor.is_floating_point() or tensor.numel() < 128:
        return False
    path_lower = str(file_path).lower()
    if any(p in path_lower for p in PROTECTED_NAMESPACES):
        return False
    key_lower = key.lower()
    if any(p in key_lower for p in PROTECTED_KEY_PATTERNS):
        return False
    return True

def quantize_clean_rtn(tensor: torch.Tensor, target_dtype: torch.dtype, compute_device: torch.device) -> torch.Tensor:
    t = tensor.to(compute_device)
    if target_dtype in [torch.bfloat16, torch.float16]:
        out = t.to(target_dtype)
    else:
        finfo = torch.finfo(target_dtype)
        out = torch.clamp(t, float(finfo.min), float(finfo.max)).to(target_dtype)
    return out.cpu()

def align_tokenizer_metadata(cache_root: Path) -> None:
    for config_path in cache_root.glob("**/tokenizer/tokenizer_config.json"):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data.get("fix_mistral_regex", False):
                data["fix_mistral_regex"] = True
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logging.info("Aligned tokenizer regex metadata: %s", config_path)
        except Exception as e:
            logging.warning("Tokenizer metadata alignment skipped for %s: %s", config_path, str(e))

def purge_unreferenced_artifacts(snapshot_path: Path) -> None:
    bloat_dir = snapshot_path / "qwen_7B"
    if bloat_dir.exists() and bloat_dir.is_dir():
        try:
            shutil.rmtree(bloat_dir)
            logging.info("Purged unreferenced training directory: %s", bloat_dir)
        except Exception as e:
            logging.warning("Failed to purge %s: %s", bloat_dir, str(e))

    safetensor_stems = {p.stem for p in snapshot_path.rglob("*.safetensors")}
    for bin_file in list(snapshot_path.rglob("*.bin")) + list(snapshot_path.rglob("*.pt")):
        if bin_file.stem in safetensor_stems or bin_file.stem.replace("pytorch_model", "model") in safetensor_stems:
            try:
                bin_file.unlink()
                logging.info("Deduplicated redundant binary container: %s", bin_file.name)
            except Exception:
                pass

def condition_model_shards(
    model_dir: Path,
    target_dtype: Optional[torch.dtype],
    quant_engine: str,
    compute_device: torch.device
) -> None:
    logging.info("Executing shard conditioning on snapshot: %s", model_dir)
    logging.info(
        "Target Precision: %s | Quant Engine: %s | Device: %s",
        str(target_dtype) if target_dtype else "NATIVE",
        quant_engine.upper(),
        str(compute_device)
    )
    purge_unreferenced_artifacts(model_dir)

    safetensor_files = list(model_dir.rglob("*.safetensors"))
    total_files = len(safetensor_files)
    if total_files == 0:
        logging.warning("No SafeTensors weights located in %s", model_dir)
        return

    processed_count = 0
    quantized_tensors_count = 0
    preserved_tensors_count = 0
    stripped_keys_count = 0

    for sf_path in safetensor_files:
        logging.info("[%d/%d] Processing Container: %s", processed_count + 1, total_files, sf_path.name)
        state_dict = load_safetensors(str(sf_path), device="cpu")
        processed_dict = {}

        for key, tensor in state_dict.items():
            if is_dead_key(key):
                stripped_keys_count += 1
                continue

            if target_dtype is not None and is_quantizable(key, sf_path, tensor):
                if quant_engine == "dto_ca" and HAS_REFINERY:
                    refined = refineTensor(
                        origin_tensor=tensor,
                        target_dtype=target_dtype,
                        fp8_dtypes=FP8_DTYPES,
                        dtype_map=DTYPE_LOOKUP,
                        original_on_disk_dtype=tensor.dtype
                    )
                    processed_dict[key] = refined.cpu()
                else:
                    processed_dict[key] = quantize_clean_rtn(tensor, target_dtype, compute_device)
                quantized_tensors_count += 1
            else:
                processed_dict[key] = tensor
                preserved_tensors_count += 1

        save_safetensors(processed_dict, str(sf_path))
        del state_dict, processed_dict
        processed_count += 1
        gc.collect()

    if compute_device.type == "cuda":
        torch.cuda.synchronize()
    if HAS_REFINERY:
        clear_refinery_cache()

    logging.info(
        "Conditioning Complete: %d quantized (%s), %d preserved native, %d dead keys removed.",
        quantized_tensors_count, quant_engine.upper(), preserved_tensors_count, stripped_keys_count
    )

def download_and_condition_weights(
    repo_id: str,
    precision: str = "native",
    quant_engine: str = "clean_rtn",
    device_str: str = "auto",
    max_workers: int = 8
) -> Path:
    logging.info("Target Cache Anchor: %s", CACHE_DIR)
    logging.info("Initiating model acquisition for: %s", repo_id)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_path_str = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        max_workers=max_workers,
        resume_download=True,
        local_dir_use_symlinks=False,
        ignore_patterns=IGNORE_DOWNLOAD_PATTERNS
    )
    snapshot_path = Path(snapshot_path_str)
    purge_unreferenced_artifacts(snapshot_path)
    align_tokenizer_metadata(CACHE_DIR)

    target_dtype = DTYPE_MAP.get(precision.lower(), None)
    if device_str == "auto":
        compute_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        compute_device = torch.device(device_str)

    condition_model_shards(snapshot_path, target_dtype, quant_engine, compute_device)
    logging.info("Acquisition and conditioning complete: %s", snapshot_path)
    return snapshot_path

def prompt_precision_menu() -> Tuple[str, str, str]:
    print("\n" + "=" * 80)
    print("           MINIMAX-MUSIC3 MODEL ACQUISITION & PRECISION SELECTOR")
    print("=" * 80)
    print(" [1] Native (None)   - Original uncompressed precision (Pristine BF16/FP32)")
    print(" [2] BF16 (Standard) - Round-to-nearest BFloat16 on Transformer blocks")
    print(" [3] FP16            - Round-to-nearest Float16 on Transformer blocks")
    print(" [4] FP8 E4M3FN      - 3-bit Mantissa FP8 (Vocoder & sensitive layers preserved)")
    print(" [5] FP8 E5M2        - 2-bit Mantissa FP8 (Vocoder & sensitive layers preserved)")
    print("-" * 80)
    choice = input("Select precision [1-5] (Default: 1): ").strip()
    mapping = {
        "1": "native", "native": "native",
        "2": "bf16", "bf16": "bf16",
        "3": "fp16", "fp16": "fp16",
        "4": "e4m3fn", "e4m3fn": "e4m3fn",
        "5": "e5m2", "e5m2": "e5m2"
    }
    precision = mapping.get(choice.lower(), "native")
    if precision == "native":
        return "native", "clean_rtn", "auto"

    quant_engine = "clean_rtn"
    if HAS_REFINERY:
        print("\n" + "-" * 80)
        print(" Select Quantization Engine:")
        print(" [1] Clean RTN        - Standard round-to-nearest (Pristine mathematical baseline)")
        print(" [2] Gorilla DTO-CA   - Gray-Code Z-Curve + Dynamic Bayer dither pass")
        print("-" * 80)
        eng_choice = input("Select engine [1-2] (Default: 1): ").strip()
        quant_engine = "dto_ca" if eng_choice == "2" else "clean_rtn"

    print("\n" + "-" * 80)
    print(" Select Compute Target for Conversion Pass:")
    print(" [1] GPU (CUDA)      - Hardware-accelerated execution")
    print(" [2] CPU (System RAM)- Zero VRAM allocation execution")
    print("-" * 80)
    dev_choice = input("Select device [1-2] (Default: 1): ").strip()
    device_str = "cpu" if dev_choice == "2" else "auto"

    return precision, quant_engine, device_str

def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and condition MiniMax-Music3 model weights.")
    parser.add_argument("--repo_id", type=str, default="MiniMaxAI/MiniMax-Music3", help="Hugging Face repository ID.")
    parser.add_argument(
        "--precision",
        "--quant_target",
        dest="precision",
        type=str,
        choices=PRECISION_CHOICES,
        default=None,
        help="Target precision for quantizable linear blocks."
    )
    parser.add_argument(
        "--engine",
        dest="quant_engine",
        type=str,
        choices=QUANT_ENGINES,
        default="clean_rtn",
        help="Quantization engine backend (clean_rtn / dto_ca)."
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Compute execution provider for conditioning pass."
    )
    parser.add_argument("--max_workers", type=int, default=8, help="Concurrent download worker threads.")
    args = parser.parse_args()

    precision = args.precision
    quant_engine = args.quant_engine
    device_str = args.device

    if precision is None:
        if sys.stdin.isatty():
            precision, quant_engine, dev_input = prompt_precision_menu()
            if device_str is None:
                device_str = dev_input
        else:
            precision = "native"

    if device_str is None:
        device_str = "auto"

    if precision in ["e4m3fn", "e5m2"] and DTYPE_MAP[precision] is None:
        logging.error("Selected precision '%s' is not supported in this PyTorch runtime.", precision)
        sys.exit(1)

    try:
        download_and_condition_weights(
            repo_id=args.repo_id,
            precision=precision,
            quant_engine=quant_engine,
            device_str=device_str,
            max_workers=args.max_workers
        )
    except Exception as e:
        logging.error("Acquisition and conditioning failure: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()