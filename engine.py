import os
import sys
import gc
import math
import time
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CACHE_DIR = ROOT_DIR / ".hf_cache"
MIOPEN_CACHE_DIR = CACHE_DIR / "miopen"
MIOPEN_DB_DIR = MIOPEN_CACHE_DIR / "db"
MIOPEN_KERNELS_DIR = MIOPEN_CACHE_DIR / "kernels"
MIOPEN_DB_DIR.mkdir(parents=True, exist_ok=True)
MIOPEN_KERNELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MIOPEN_USER_DB_PATH"] = str(MIOPEN_DB_DIR)
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = str(MIOPEN_KERNELS_DIR)
os.environ["MIOPEN_FIND_MODE"] = "2"
os.environ["MIOPEN_LOG_LEVEL"] = "0"
os.environ["MIOPEN_ENABLE_LOGGING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import numpy as np
import soundfile as sf
import torch
import torch.fft
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

try:
    from transformers import Qwen3ForCausalLM, Qwen3Config
except ImportError:
    Qwen3ForCausalLM = AutoModelForCausalLM
    Qwen3Config = AutoConfig

from safetensors.torch import load_file as load_safetensors

from models.depth_decoder import MiniMaxMusic3RVQDepthDecoder
from models.condition_encoder import MiniMaxMusic3ConditionEncoder
from models.transformer import MiniMaxMusic3Transformer1DModel
from models.vocoder import MiniMaxMusic3Vocoder
from pipeline.prompt_compiler import build_text_ids
from pipeline.schedulers import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
)
from pipeline.music_pipeline import MiniMaxMusic3Pipeline
from schema import GenerationRequest, GenerationResponse


def fold_weight_norm(module: torch.nn.Module) -> None:
    for sub_module in module.modules():
        try:
            torch.nn.utils.remove_weight_norm(sub_module)
        except (ValueError, AttributeError):
            pass


def load_sharded_safetensors(
    model_dir: Path, target_module: torch.nn.Module, device: torch.device, dtype: torch.dtype
) -> None:
    state_dict = {}
    sf_files = sorted(list(set(model_dir.rglob("*.safetensors"))))
    if sf_files:
        for sf_path in sf_files:
            state_dict.update(load_safetensors(str(sf_path), device="cpu"))
    else:
        bin_files = sorted(list(set(model_dir.rglob("*.bin"))) | set(model_dir.rglob("*.pt")))
        for bf in bin_files:
            state_dict.update(torch.load(str(bf), map_location="cpu", weights_only=True))

    has_weight_g = any("weight_g" in k for k in state_dict.keys())
    if not has_weight_g:
        fold_weight_norm(target_module)
        target_module.load_state_dict(state_dict, strict=True)
    else:
        target_module.load_state_dict(state_dict, strict=True)
        fold_weight_norm(target_module)

    target_module.to(device=device, dtype=dtype)


def resolve_model_path(repo_or_path: str) -> Path:
    direct_path = Path(repo_or_path)
    if direct_path.exists():
        return direct_path

    hub_path = CACHE_DIR / "hub"
    repo_folder_name = "models--" + repo_or_path.replace("/", "--")
    candidate = hub_path / repo_folder_name / "snapshots"
    if candidate.exists():
        snapshots = list(candidate.iterdir())
        if snapshots:
            return snapshots[0]
    return direct_path


def apply_dct_2(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    N = x.shape[dim]
    if dim != -1 and dim != x.ndim - 1:
        x = x.transpose(dim, -1)
    orig_shape = x.shape
    x_2d = x.reshape(-1, N)

    idx = torch.empty(N, dtype=torch.long, device=x.device)
    idx[: (N + 1) // 2] = torch.arange(0, N, 2, device=x.device)
    idx[(N + 1) // 2 :] = torch.arange(N - 1 - (N % 2), 0, -2, device=x.device)
    v = x_2d[:, idx]
    V = torch.fft.fft(v, dim=-1)

    k = torch.arange(N, dtype=torch.float32, device=x.device)
    angles = -math.pi * k / (2.0 * N)
    rot = torch.complex(torch.cos(angles), torch.sin(angles))
    X_raw = (V * rot).real

    scale = torch.full((N,), math.sqrt(2.0 / N), dtype=torch.float32, device=x.device)
    scale[0] = math.sqrt(1.0 / N)
    X = X_raw * scale
    out = X.reshape(orig_shape)
    if dim != -1 and dim != orig_shape.__len__() - 1:
        out = out.transpose(dim, -1)
    return out


def apply_idct_2(X: torch.Tensor, dim: int = -1) -> torch.Tensor:
    N = X.shape[dim]
    if dim != -1 and dim != X.ndim - 1:
        X = X.transpose(dim, -1)
    orig_shape = X.shape
    X_2d = X.reshape(-1, N)

    scale = torch.full((N,), math.sqrt(2.0 / N), dtype=torch.float32, device=X.device)
    scale[0] = math.sqrt(1.0 / N)
    X_unnorm = X_2d / scale

    X_ext = torch.zeros((X_2d.shape[0], N), dtype=torch.float32, device=X.device)
    X_ext[:, 1:] = X_unnorm[:, 1:].flip(dims=[-1])
    V_complex = torch.complex(X_unnorm, -X_ext)
    V_complex[:, 0] = X_unnorm[:, 0]

    k = torch.arange(N, dtype=torch.float32, device=X.device)
    angles = math.pi * k / (2.0 * N)
    rot = torch.complex(torch.cos(angles), torch.sin(angles))
    V = V_complex * rot
    v = torch.fft.ifft(V, dim=-1).real

    x_out = torch.empty_like(v)
    half = (N + 1) // 2
    x_out[:, 0::2] = v[:, :half]
    x_out[:, 1::2] = v[:, half:].flip(dims=[-1])
    out = x_out.reshape(orig_shape)
    if dim != -1 and dim != orig_shape.__len__() - 1:
        out = out.transpose(dim, -1)
    return out


def apply_per_channel_blue_noise(
    tensor: torch.Tensor,
    alpha: float = 0.75,
    floor_eps: float = 0.40,
    blend_homotopy: float = 1.0,
) -> torch.Tensor:
    orig_dtype = tensor.dtype
    work_tensor = tensor.to(dtype=torch.float32)
    orig_shape = work_tensor.shape
    N = orig_shape[-1]
    work_2d = work_tensor.reshape(-1, N)

    spectrum = apply_dct_2(work_2d, dim=-1)
    k = torch.arange(N, device=tensor.device, dtype=torch.float32)
    norm_freq = k / float(max(N - 1, 1))
    H_k = torch.pow(floor_eps + (1.0 - floor_eps) * norm_freq, alpha)
    gamma_base = math.sqrt(float(N) / float(torch.sum(H_k**2).clamp(min=1e-8).item()))

    theta = (math.pi / 2.0) * min(max(blend_homotopy, 0.0), 1.0)
    G_k = math.cos(theta) + math.sin(theta) * (gamma_base * H_k)
    gamma_theta = math.sqrt(float(N) / float(torch.sum(G_k**2).clamp(min=1e-8).item()))

    filter_kernel = gamma_theta * G_k
    filtered_spectrum = spectrum * filter_kernel
    out_2d = apply_idct_2(filtered_spectrum, dim=-1)
    out_tensor = out_2d.reshape(orig_shape)
    return out_tensor.to(dtype=orig_dtype)


def apply_temporal_perona_malik_pde(
    tensor: torch.Tensor,
    iterations: int = 5,
    conductance: float = 0.15,
    stability_lambda: float = 0.20,
    is_blue_noise: bool = False,
    blue_noise_alpha: float = 0.75,
) -> torch.Tensor:
    orig_dtype = tensor.dtype
    work_tensor = tensor.to(dtype=torch.float32)
    if is_blue_noise:
        k_eff = conductance * math.sqrt(1.0 + 2.0 * (blue_noise_alpha**2))
    else:
        k_eff = conductance
    k_sq = max(k_eff**2, 1e-8)

    orig_shape = work_tensor.shape
    u = work_tensor.reshape(-1, 1, orig_shape[-1])
    orig_mean = u.mean(dim=-1, keepdim=True)
    orig_std = u.std(dim=-1, keepdim=True).clamp(min=1e-8)
    u_diff = u.clone()

    for _ in range(iterations):
        grad_east = torch.zeros_like(u_diff)
        grad_west = torch.zeros_like(u_diff)
        grad_east[:, :, :-1] = u_diff[:, :, 1:] - u_diff[:, :, :-1]
        grad_west[:, :, 1:] = u_diff[:, :, :-1] - u_diff[:, :, 1:]
        c_east = torch.exp(-(grad_east**2) / k_sq)
        c_west = torch.exp(-(grad_west**2) / k_sq)
        divergence = c_east * grad_east + c_west * grad_west
        u_diff = u_diff + stability_lambda * divergence

    diff_mean = u_diff.mean(dim=-1, keepdim=True)
    diff_std = u_diff.std(dim=-1, keepdim=True).clamp(min=1e-8)
    u_standardized = orig_mean + (u_diff - diff_mean) * (orig_std / diff_std)
    out_tensor = u_standardized.reshape(orig_shape)
    return out_tensor.to(dtype=orig_dtype)


def apply_sub_millisecond_declick(audio_tensor: torch.Tensor, fade_samples: int = 512) -> torch.Tensor:
    if audio_tensor.shape[-1] <= fade_samples * 2:
        return audio_tensor
    fade = 0.5 * (
        1.0
        - torch.cos(
            torch.linspace(0.0, math.pi, fade_samples, device=audio_tensor.device, dtype=audio_tensor.dtype)
        )
    )
    audio_tensor[..., :fade_samples] *= fade
    audio_tensor[..., -fade_samples:] *= torch.flip(fade, dims=[0])
    return audio_tensor


class MusicEngine:
    def __init__(
        self,
        repo_id: str = "MiniMaxAI/MiniMax-Music3",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.repo_id = repo_id
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.pipeline: Optional[MiniMaxMusic3Pipeline] = None
        self._current_offload_state: Optional[bool] = None

    def _init_components(self, cpu_offload: bool) -> None:
        if self.pipeline is not None and self._current_offload_state == cpu_offload:
            return

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA execution requested but no compatible device detected.")

        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        root_path = resolve_model_path(self.repo_id)
        target_device = torch.device("cpu") if cpu_offload else self.device

        tokenizer_dir = root_path / "tokenizer" if (root_path / "tokenizer").exists() else root_path
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)

        lm_dir = root_path / "language_model" if (root_path / "language_model").exists() else root_path
        lm_config_path = lm_dir / "config.json"
        if lm_config_path.exists():
            with open(lm_config_path, "r", encoding="utf-8") as f:
                lm_cfg_dict = json.load(f)
            lm_config = Qwen3Config(**lm_cfg_dict)
        else:
            lm_config = Qwen3Config()

        language_model = Qwen3ForCausalLM(lm_config)
        load_sharded_safetensors(lm_dir, language_model, target_device, self.dtype)
        language_model.eval()

        rvq_dir = root_path / "rvq_depth_decoder" if (root_path / "rvq_depth_decoder").exists() else root_path
        rvq_depth_decoder = MiniMaxMusic3RVQDepthDecoder()
        load_sharded_safetensors(rvq_dir, rvq_depth_decoder, target_device, self.dtype)
        fold_weight_norm(rvq_depth_decoder)
        rvq_depth_decoder.eval()

        cond_dir = root_path / "condition_encoder" if (root_path / "condition_encoder").exists() else root_path
        condition_encoder = MiniMaxMusic3ConditionEncoder()
        load_sharded_safetensors(cond_dir, condition_encoder, target_device, self.dtype)
        condition_encoder.eval()

        transformer_dir = root_path / "transformer" if (root_path / "transformer").exists() else root_path
        transformer = MiniMaxMusic3Transformer1DModel()
        load_sharded_safetensors(transformer_dir, transformer, target_device, self.dtype)
        transformer.eval()

        vocoder_dir = root_path / "vocoder" if (root_path / "vocoder").exists() else root_path
        vocoder = MiniMaxMusic3Vocoder()
        load_sharded_safetensors(vocoder_dir, vocoder, target_device, torch.float32)
        fold_weight_norm(vocoder)
        vocoder.eval()

        self.pipeline = MiniMaxMusic3Pipeline(
            tokenizer=tokenizer,
            language_model=language_model,
            rvq_depth_decoder=rvq_depth_decoder,
            condition_encoder=condition_encoder,
            transformer=transformer,
            vocoder=vocoder,
            sampling_rate=vocoder.sampling_rate,
        )
        self._current_offload_state = cpu_offload

    def _compute_dynamic_shift(self, audio_duration: float, sampling_rate: int = 44100) -> float:
        base_shift = 0.5
        max_shift = 1.15
        base_seq_len = 256
        max_seq_len = 4096
        latent_seq_len = int(math.ceil((audio_duration * sampling_rate) / 512))
        ratio = max(0.0, min(1.0, (latent_seq_len - base_seq_len) / float(max_seq_len - base_seq_len)))
        return base_shift + ratio * (max_shift - base_shift)

    def synthesize(self, request: GenerationRequest) -> GenerationResponse:
        request.validate()
        self._init_components(request.cpu_offload)

        effective_prompt = request.compile_prompt()
        sanitized_lyrics = request.sanitize_lyrics()
        sampling_rate = self.pipeline.sampling_rate

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.perf_counter()

        if request.seed is not None and request.seed >= 0:
            torch.manual_seed(request.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request.seed)
            np.random.seed(request.seed % (2**32))
            generator = torch.Generator(device=self.device).manual_seed(request.seed)
        else:
            generator = None

        text_device = self.device if not request.cpu_offload else self.device
        text_ids = build_text_ids(
            self.pipeline.tokenizer,
            effective_prompt,
            sanitized_lyrics,
            device=text_device,
        )

        if request.cpu_offload:
            self.pipeline.language_model.to(self.device)
            self.pipeline.rvq_depth_decoder.to(self.device)

        frame_hiddens = self.pipeline.generate_stage1_autoregressive(
            text_ids=text_ids,
            audio_duration=request.audio_duration,
            generator=generator,
            cfg_scale=1.5,
            cfg_top_k=request.top_k if request.top_k is not None else 50,
            sampling_top_k=request.top_k if request.top_k is not None else 50,
            top_p=request.top_p,
            temperature=request.temperature,
            show_progress=True,
        )

        if request.cpu_offload:
            self.pipeline.language_model.to("cpu")
            self.pipeline.rvq_depth_decoder.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.pipeline.condition_encoder.to(self.device)
            self.pipeline.transformer.to(self.device)

        dynamic_shift = self._compute_dynamic_shift(request.audio_duration, sampling_rate)
        if request.scheduler_type == "heun":
            scheduler = FlowMatchHeunDiscreteScheduler(shift=dynamic_shift)
        else:
            scheduler = FlowMatchEulerDiscreteScheduler(shift=dynamic_shift)

        def latent_shaping_fn(latents: torch.Tensor) -> torch.Tensor:
            out = latents
            is_blue = (request.noise_topology == "blue_noise")
            if is_blue:
                out = apply_per_channel_blue_noise(
                    out,
                    alpha=request.blue_noise_alpha,
                    floor_eps=0.40,
                    blend_homotopy=1.0,
                )
            if request.enable_pm_diffusion:
                out = apply_temporal_perona_malik_pde(
                    out,
                    iterations=request.pm_iterations,
                    conductance=request.pm_conductance,
                    stability_lambda=request.pm_lambda,
                    is_blue_noise=is_blue,
                    blue_noise_alpha=request.blue_noise_alpha,
                )
            return out

        latent_chunks = self.pipeline.generate_stage2_flow_matching(
            frame_hiddens=frame_hiddens,
            scheduler=scheduler,
            num_inference_steps=request.num_inference_steps if request.num_inference_steps is not None else 30,
            guidance_scale=request.guidance_scale if request.guidance_scale is not None else 1.7,
            generator=generator,
            latent_shaping_fn=latent_shaping_fn,
            device=self.device,
            show_progress=True,
        )

        if request.cpu_offload:
            self.pipeline.condition_encoder.to("cpu")
            self.pipeline.transformer.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.pipeline.vocoder.to(self.device)

        audio_tensor = self.pipeline.decode_latents(latent_chunks)

        if request.cpu_offload:
            self.pipeline.vocoder.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if audio_tensor.ndim == 3:
            audio_tensor = audio_tensor.squeeze(0)

        if request.apply_declick:
            audio_tensor = apply_sub_millisecond_declick(audio_tensor, fade_samples=512)

        elapsed_time = time.perf_counter() - start_time
        peak_vram_gb = 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)

        peak_val = torch.max(torch.abs(audio_tensor)).item()
        rms_val = torch.sqrt(torch.mean(audio_tensor**2)).item()
        peak_dbfs = 20.0 * math.log10(max(peak_val, 1e-12))
        rms_dbfs = 20.0 * math.log10(max(rms_val, 1e-12))
        crest_factor_db = peak_dbfs - rms_dbfs

        audio_data = audio_tensor.detach().cpu().numpy()
        if audio_data.shape[0] < audio_data.shape[1]:
            audio_data = audio_data.T
        audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)

        out_path = Path(request.output_path)
        if not out_path.is_absolute():
            out_path = ROOT_DIR / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio_data, sampling_rate, subtype="FLOAT")

        total_samples = audio_data.shape[0]
        actual_duration = total_samples / float(sampling_rate)
        rtf = elapsed_time / max(actual_duration, 1e-6)

        return GenerationResponse(
            output_path=str(out_path),
            sample_rate=sampling_rate,
            total_samples=total_samples,
            duration_seconds=actual_duration,
            generation_time_seconds=elapsed_time,
            real_time_factor=rtf,
            peak_linear=peak_val,
            peak_dbfs=peak_dbfs,
            rms_dbfs=rms_dbfs,
            crest_factor_db=crest_factor_db,
            scheduler_used=request.scheduler_type,
            noise_topology_used=request.noise_topology,
            pm_diffusion_used=request.enable_pm_diffusion,
            effective_prompt=effective_prompt,
            declick_applied=request.apply_declick,
            cpu_offload_active=bool(self._current_offload_state),
            peak_vram_gb=peak_vram_gb,
        )