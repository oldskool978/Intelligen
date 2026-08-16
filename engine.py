import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CACHE_DIR = ROOT_DIR / ".hf_cache"

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MIOPEN_LOG_LEVEL"] = "0"
os.environ["MIOPEN_ENABLE_LOGGING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import math
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*There are modules in.*should be kept in float32.*")
warnings.filterwarnings("ignore", message=".*Modular Diffusers is currently an experimental feature.*")
warnings.filterwarnings("ignore", message=".*Guiders are currently an experimental feature.*")

import numpy as np
import soundfile as sf
import torch
import torch.fft
from diffusers import (
    ModularPipeline,
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
)

from schema import GenerationRequest, GenerationResponse

SCHEDULER_REGISTRY = {
    "euler": FlowMatchEulerDiscreteScheduler,
    "heun": FlowMatchHeunDiscreteScheduler,
}

def apply_blue_noise_dispersion(tensor: torch.Tensor, alpha: float) -> torch.Tensor:
    n_fft = tensor.shape[-1]
    spectrum = torch.fft.rfft(tensor, n=n_fft, dim=-1)
    k = torch.arange(spectrum.shape[-1], device=tensor.device, dtype=torch.float32)
    norm_freq = k / float(max(spectrum.shape[-1] - 1, 1))
    
    spectral_filter = torch.pow(torch.clamp(norm_freq, min=1e-4), alpha)
    spectral_filter[0] = 0.0
    
    filtered_spectrum = spectrum * spectral_filter
    filtered_tensor = torch.fft.irfft(filtered_spectrum, n=n_fft, dim=-1)
    
    mean = torch.mean(filtered_tensor)
    std = torch.std(filtered_tensor)
    return (filtered_tensor - mean) / torch.clamp(std, min=1e-8)

def apply_perona_malik_diffusion(
    tensor: torch.Tensor,
    iterations: int,
    conductance: float,
    stability_lambda: float
) -> torch.Tensor:
    u = tensor.clone()
    k_sq = conductance ** 2
    
    for _ in range(iterations):
        grad_east = torch.zeros_like(u)
        grad_west = torch.zeros_like(u)
        grad_south = torch.zeros_like(u)
        grad_north = torch.zeros_like(u)
        
        grad_east[..., :, :-1] = u[..., :, 1:] - u[..., :, :-1]
        grad_west[..., :, 1:] = u[..., :, :-1] - u[..., :, 1:]
        grad_south[..., :-1, :] = u[..., 1:, :] - u[..., :-1, :]
        grad_north[..., 1:, :] = u[..., :-1, :] - u[..., 1:, :]
        
        c_east = torch.exp(- (grad_east ** 2) / k_sq)
        c_west = torch.exp(- (grad_west ** 2) / k_sq)
        c_south = torch.exp(- (grad_south ** 2) / k_sq)
        c_north = torch.exp(- (grad_north ** 2) / k_sq)
        
        divergence = (
            c_east * grad_east +
            c_west * grad_west +
            c_south * grad_south +
            c_north * grad_north
        )
        u = u + stability_lambda * divergence
        
    mean = torch.mean(u)
    std = torch.std(u)
    return (u - mean) / torch.clamp(std, min=1e-8)

class MusicEngine:
    def __init__(
        self,
        repo_id: str = "MiniMaxAI/MiniMax-Music3",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        self.repo_id = repo_id
        self.device = device
        self.dtype = dtype
        self.pipe: Optional[ModularPipeline] = None
        self._initialize_pipeline()

    def _fold_weight_norm_hooks(self, module: torch.nn.Module) -> None:
        for sub_module in module.modules():
            try:
                torch.nn.utils.remove_weight_norm(sub_module)
            except (ValueError, AttributeError):
                pass

    def _set_pipeline_eval(self) -> None:
        components = getattr(self.pipe, "components", {})
        if isinstance(components, dict):
            for component in components.values():
                if isinstance(component, torch.nn.Module):
                    component.eval()
        for attr_name in dir(self.pipe):
            if not attr_name.startswith("_"):
                try:
                    attr = getattr(self.pipe, attr_name)
                    if isinstance(attr, torch.nn.Module):
                        attr.eval()
                except Exception:
                    pass

    def _initialize_pipeline(self) -> None:
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA target requested but no compatible GPU was detected.")

        self.pipe = ModularPipeline.from_pretrained(
            self.repo_id,
            local_files_only=True
        )

        self.pipe.load_components(
            dtype=self.dtype,
            local_files_only=True
        )

        if hasattr(self.pipe, "vocoder") and isinstance(self.pipe.vocoder, torch.nn.Module):
            self._fold_weight_norm_hooks(self.pipe.vocoder)
        if hasattr(self.pipe, "rvq_depth_decoder") and isinstance(self.pipe.rvq_depth_decoder, torch.nn.Module):
            self._fold_weight_norm_hooks(self.pipe.rvq_depth_decoder)

        self.pipe.to(self.device)
        self._set_pipeline_eval()

    def _configure_scheduler(
        self,
        scheduler_type: str,
        audio_duration: float,
        sampling_rate: int,
        upsampling_factor: int = 512
    ) -> None:
        if hasattr(self.pipe, "scheduler") and self.pipe.scheduler is not None:
            config = dict(getattr(self.pipe.scheduler, "config", {}))
            base_shift = config.get("base_shift", 0.5)
            max_shift = config.get("max_shift", 1.15)
            base_seq_len = config.get("base_image_seq_len", 256)
            max_seq_len = config.get("max_image_seq_len", 4096)
            
            latent_seq_len = int(math.ceil((audio_duration * sampling_rate) / upsampling_factor))
            ratio = max(0.0, min(1.0, (latent_seq_len - base_seq_len) / float(max_seq_len - base_seq_len)))
            dynamic_shift = base_shift + ratio * (max_shift - base_shift)
            
            if scheduler_type == "native" or scheduler_type not in SCHEDULER_REGISTRY:
                scheduler_cls = self.pipe.scheduler.__class__
            else:
                scheduler_cls = SCHEDULER_REGISTRY[scheduler_type]

            new_scheduler = scheduler_cls.from_config(
                config,
                shift=dynamic_shift,
                use_dynamic_shifting=False
            )
            self.pipe.scheduler = new_scheduler
            if hasattr(self.pipe, "components") and isinstance(self.pipe.components, dict):
                self.pipe.components["scheduler"] = new_scheduler

    def _configure_guidance(self, guidance_scale: Optional[float]) -> None:
        if guidance_scale is None:
            return
        if hasattr(self.pipe, "guider") and self.pipe.guider is not None:
            if hasattr(self.pipe.guider, "guidance_scale"):
                self.pipe.guider.guidance_scale = guidance_scale
            elif hasattr(self.pipe.guider, "config") and isinstance(self.pipe.guider.config, dict):
                self.pipe.guider.config["guidance_scale"] = guidance_scale

    def _configure_autoregressive_sampling(
        self,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int]
    ) -> None:
        if temperature is None and top_p is None and top_k is None:
            return

        candidate_modules = []
        components = getattr(self.pipe, "components", {})
        if isinstance(components, dict):
            candidate_modules.extend(components.values())

        for mod in candidate_modules:
            if mod is None:
                continue
            if hasattr(mod, "generation_config") and mod.generation_config is not None:
                if temperature is not None:
                    mod.generation_config.temperature = temperature
                if top_p is not None:
                    mod.generation_config.top_p = top_p
                if top_k is not None:
                    mod.generation_config.top_k = top_k
                mod.generation_config.do_sample = True

    def _generate_conditioned_latents(
        self,
        request: GenerationRequest,
        sampling_rate: int,
        upsampling_factor: int = 512
    ) -> Optional[torch.Tensor]:
        if request.noise_topology == "gaussian":
            return None
            
        latent_channels = 64
        if hasattr(self.pipe, "transformer") and hasattr(self.pipe.transformer, "config"):
            latent_channels = getattr(self.pipe.transformer.config, "in_channels", latent_channels)
            
        latent_seq_len = int(math.ceil((request.audio_duration * sampling_rate) / upsampling_factor))
        shape = (1, latent_channels, latent_seq_len)
        
        gen = torch.Generator(device="cpu").manual_seed(request.seed) if request.seed is not None and request.seed >= 0 else None
        base_noise = torch.randn(shape, generator=gen, dtype=torch.float32, device="cpu").to(self.device)
        
        if request.noise_topology == "blue_noise":
            shaped_latents = apply_blue_noise_dispersion(base_noise, alpha=request.blue_noise_alpha)
        elif request.noise_topology == "perona_malik":
            shaped_latents = apply_perona_malik_diffusion(
                base_noise,
                iterations=request.pm_iterations,
                conductance=request.pm_conductance,
                stability_lambda=request.pm_lambda
            )
        else:
            shaped_latents = base_noise
            
        return shaped_latents.to(dtype=self.dtype)

    def synthesize(self, request: GenerationRequest) -> GenerationResponse:
        request.validate()
        effective_prompt = request.compile_prompt()
        sanitized_lyrics = request.sanitize_lyrics()

        sampling_rate = getattr(self.pipe, "sampling_rate", None)
        if sampling_rate is None and hasattr(self.pipe, "vocoder"):
            sampling_rate = getattr(self.pipe.vocoder, "config", {}).get("sampling_rate", 44100)
        if sampling_rate is None:
            sampling_rate = 44100

        self._configure_scheduler(
            scheduler_type=request.scheduler_type,
            audio_duration=request.audio_duration,
            sampling_rate=sampling_rate,
            upsampling_factor=512
        )

        self._configure_guidance(guidance_scale=request.guidance_scale)
        self._configure_autoregressive_sampling(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )

        generator = (
            torch.Generator(self.device).manual_seed(request.seed)
            if request.seed is not None and request.seed >= 0
            else None
        )

        custom_latents = self._generate_conditioned_latents(
            request=request,
            sampling_rate=sampling_rate,
            upsampling_factor=512
        )

        pipeline_kwargs = {
            "prompt": effective_prompt,
            "lyrics": sanitized_lyrics,
            "audio_duration": request.audio_duration,
            "generator": generator,
            "output": "audios",
        }

        if request.num_inference_steps is not None:
            pipeline_kwargs["num_inference_steps"] = request.num_inference_steps

        if custom_latents is not None:
            pipeline_kwargs["latents"] = custom_latents

        start_time = time.perf_counter()

        with torch.inference_mode():
            raw_output = self.pipe(**pipeline_kwargs)[0]

        elapsed_time = time.perf_counter() - start_time

        if isinstance(raw_output, torch.Tensor):
            audio_tensor = raw_output.to(device=self.device, dtype=torch.float32)
        else:
            audio_tensor = torch.as_tensor(raw_output, device=self.device, dtype=torch.float32)

        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.ndim == 3:
            audio_tensor = audio_tensor.squeeze(0)

        peak_val = torch.max(torch.abs(audio_tensor)).item()
        rms_val = torch.sqrt(torch.mean(audio_tensor ** 2)).item()
        peak_dbfs = 20.0 * math.log10(max(peak_val, 1e-12))
        rms_dbfs = 20.0 * math.log10(max(rms_val, 1e-12))

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
            scheduler_used=request.scheduler_type,
            noise_topology_used=request.noise_topology,
            effective_prompt=effective_prompt
        )