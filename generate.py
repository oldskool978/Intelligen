import os
import sys
import math
import warnings
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / ".hf_cache"

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MIOPEN_LOG_LEVEL"] = "0"
os.environ["MIOPEN_ENABLE_LOGGING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*Modular Diffusers is currently an experimental feature.*")
warnings.filterwarnings("ignore", message=".*Guiders are currently an experimental feature.*")

import numpy as np
import soundfile as sf
import torch
from diffusers import ModularPipeline

DEFAULT_LYRICS = """[intro]
[verse]
Morning sunlight breaks across the bay
Chasing all the shadow forms away
[chorus]
We are sailing where the rhythm flows
Every heartbeat in the undertow
[outro]
"""

DEFAULT_PROMPT = (
    "Genre: Synthwave Pop. Subgenre: Retrowave 80s Dance. BPM: 118. Key: A minor. Mood: Nostalgic, euphoric, driving. "
    "Vocals: Crisp male lead vocal, energetic delivery, centered mix, stacked 80s octave harmonies and gated reverb on chorus. "
    "Arrangement: Punchy analog bass synthesizer, LinnDrum gated snare and kick, lush Juno-106 analog pads, sidechained pumping, arpeggiated lead synth riff."
)


def fold_weight_norm_hooks(module: torch.nn.Module) -> None:
    for sub_module in module.modules():
        try:
            torch.nn.utils.remove_weight_norm(sub_module)
        except (ValueError, AttributeError):
            pass


def _get_module(pipe: ModularPipeline, name: str) -> Optional[torch.nn.Module]:
    mod = getattr(pipe, name, None)
    if mod is None and hasattr(pipe, "components") and isinstance(pipe.components, dict):
        mod = pipe.components.get(name, None)
    return mod if isinstance(mod, torch.nn.Module) else None


def _cast_inputs_to_fp32(module: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    new_args = tuple(
        a.to(dtype=torch.float32) if isinstance(a, torch.Tensor) and a.is_floating_point() else a
        for a in args
    )
    new_kwargs = {
        k: (v.to(dtype=torch.float32) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
        for k, v in kwargs.items()
    }
    return new_args, new_kwargs


def set_pipeline_eval(pipe: ModularPipeline) -> None:
    components = getattr(pipe, "components", {})
    if isinstance(components, dict):
        for component in components.values():
            if isinstance(component, torch.nn.Module):
                component.eval()
    for attr_name in dir(pipe):
        if not attr_name.startswith("_"):
            try:
                attr = getattr(pipe, attr_name)
                if isinstance(attr, torch.nn.Module):
                    attr.eval()
            except Exception:
                pass


def configure_flow_scheduler_shift(pipe: ModularPipeline, audio_duration: float, sampling_rate: int = 32000, upsampling_factor: int = 512) -> None:
    if hasattr(pipe, "scheduler") and pipe.scheduler is not None:
        config = dict(getattr(pipe.scheduler, "config", {}))
        base_shift = config.get("base_shift", 0.5)
        max_shift = config.get("max_shift", 1.15)
        base_seq_len = config.get("base_image_seq_len", 256)
        max_seq_len = config.get("max_image_seq_len", 4096)
        
        latent_seq_len = int(math.ceil((audio_duration * sampling_rate) / upsampling_factor))
        ratio = max(0.0, min(1.0, (latent_seq_len - base_seq_len) / float(max_seq_len - base_seq_len)))
        dynamic_shift = base_shift + ratio * (max_shift - base_shift)
        
        scheduler_cls = pipe.scheduler.__class__
        new_scheduler = scheduler_cls.from_config(
            config,
            shift=dynamic_shift,
            use_dynamic_shifting=False
        )
        pipe.scheduler = new_scheduler
        if hasattr(pipe, "components") and isinstance(pipe.components, dict):
            pipe.components["scheduler"] = new_scheduler


def run_synthesis(
    repo_id: str,
    prompt: str,
    lyrics: str,
    audio_duration: float,
    seed: int,
    output_path: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA execution provider requested but no compatible device detected.")

    pipe = ModularPipeline.from_pretrained(
        repo_id,
        local_files_only=True
    )

    pipe.load_components(
        dtype=dtype,
        local_files_only=True
    )

    rvq_mod = _get_module(pipe, "rvq_depth_decoder")
    if rvq_mod is not None:
        fold_weight_norm_hooks(rvq_mod)
        rvq_mod.to(device=device, dtype=dtype)

    lm_mod = _get_module(pipe, "language_model")
    if lm_mod is not None:
        lm_mod.to(device=device, dtype=dtype)

    transformer_mod = _get_module(pipe, "transformer")
    if transformer_mod is not None:
        transformer_mod.to(device=device, dtype=dtype)

    vocoder_mod = _get_module(pipe, "vocoder")
    if vocoder_mod is not None:
        fold_weight_norm_hooks(vocoder_mod)
        vocoder_mod.to(device=device, dtype=torch.float32)
        try:
            vocoder_mod.register_forward_pre_hook(_cast_inputs_to_fp32, with_kwargs=True)
        except TypeError:
            vocoder_mod.register_forward_pre_hook(
                lambda m, a: tuple(x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in a)
            )

    audio_vae_mod = _get_module(pipe, "audio_vae")
    if audio_vae_mod is not None:
        fold_weight_norm_hooks(audio_vae_mod)
        audio_vae_mod.to(device=device, dtype=torch.float32)
        try:
            audio_vae_mod.register_forward_pre_hook(_cast_inputs_to_fp32, with_kwargs=True)
        except TypeError:
            audio_vae_mod.register_forward_pre_hook(
                lambda m, a: tuple(x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in a)
            )

    pipe.to(device)
    set_pipeline_eval(pipe)

    sampling_rate = getattr(pipe, "sampling_rate", None)
    if sampling_rate is None and hasattr(pipe, "vocoder"):
        sampling_rate = getattr(pipe.vocoder, "config", {}).get("sampling_rate", 32000)
    if sampling_rate is None:
        sampling_rate = 32000

    configure_flow_scheduler_shift(pipe, audio_duration=audio_duration, sampling_rate=sampling_rate, upsampling_factor=512)

    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

    with torch.inference_mode():
        raw_output = pipe(
            prompt=prompt,
            lyrics=lyrics,
            audio_duration=audio_duration,
            generator=generator,
            output="audios",
        )[0]

    if isinstance(raw_output, torch.Tensor):
        audio_tensor = raw_output.to(device=device, dtype=torch.float32)
    else:
        audio_tensor = torch.as_tensor(raw_output, device=device, dtype=torch.float32)

    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    elif audio_tensor.ndim == 3:
        audio_tensor = audio_tensor.squeeze(0)

    audio_data = audio_tensor.detach().cpu().numpy()
    if audio_data.shape[0] < audio_data.shape[1]:
        audio_data = audio_data.T

    audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio_data, sampling_rate, subtype="FLOAT")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic neural music generation with MiniMax-Music3.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="MiniMaxAI/MiniMax-Music3",
        help="Repository identifier in localized cache."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Musical arrangement and stylistic prompt."
    )
    parser.add_argument(
        "--lyrics",
        type=str,
        default=DEFAULT_LYRICS,
        help="Structured lyrical markup."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=45.0,
        help="Generated track length in seconds."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed for deterministic synthesis."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="minimax_song.wav",
        help="Destination audio path relative to project root."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute execution target (cuda/cpu)."
    )

    args = parser.parse_args()
    target_output = ROOT_DIR / args.output if not Path(args.output).is_absolute() else Path(args.output)

    try:
        run_synthesis(
            repo_id=args.repo_id,
            prompt=args.prompt,
            lyrics=args.lyrics,
            audio_duration=args.duration,
            seed=args.seed,
            output_path=target_output,
            device=args.device,
            dtype=torch.bfloat16
        )
    except Exception as e:
        print(f"Generation failure: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()