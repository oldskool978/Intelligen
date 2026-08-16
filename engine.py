# engine.py
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
import copy
import warnings
from typing import Optional, Dict, Any, Union, List

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*There are modules in.*should be kept in float32.*")
warnings.filterwarnings("ignore", message=".*Modular Diffusers is currently an experimental feature.*")
warnings.filterwarnings("ignore", message=".*Guiders are currently an experimental feature.*")

import numpy as np
import soundfile as sf
import torch
import torch.fft
from diffusers import ModularPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput

from schema import GenerationRequest, GenerationResponse

class MiniMaxFlowMatchHeunDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """
    2nd-Order Predictor-Corrector (Heun / Improved Euler) Flow Matching Solver.
    Full compatibility with MiniMax-Music3 dynamic sigmas and modular chunking pipelines.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_i = None
        self._v1 = None
        self._h = None
        self._step_index = 0

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        sigmas: Optional[Union[List[float], torch.Tensor]] = None,
        mu: Optional[float] = None,
    ) -> None:
        super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu
        )

        base_sigmas = self.sigmas
        base_timesteps = self.timesteps

        num_intervals = len(base_sigmas) - 1
        if num_intervals <= 0:
            return

        scale = base_timesteps[0] / base_sigmas[0] if base_sigmas[0] != 0 else torch.tensor(1.0, device=base_sigmas.device)
        terminal_timestep = base_sigmas[-1] * scale
        full_timesteps = torch.cat([base_timesteps, terminal_timestep.unsqueeze(0)])

        heun_timesteps = []
        heun_sigmas = []

        for i in range(num_intervals):
            t_curr = full_timesteps[i]
            t_next = full_timesteps[i + 1]
            s_curr = base_sigmas[i]
            s_next = base_sigmas[i + 1]

            heun_timesteps.extend([t_curr, t_next])
            heun_sigmas.extend([s_curr, s_next])

        heun_sigmas.append(base_sigmas[-1])

        self.timesteps = (
            torch.stack(heun_timesteps)
            if isinstance(heun_timesteps[0], torch.Tensor)
            else torch.tensor(heun_timesteps, device=device)
        )
        self.sigmas = (
            torch.stack(heun_sigmas)
            if isinstance(heun_sigmas[0], torch.Tensor)
            else torch.tensor(heun_sigmas, device=device)
        )

        self._step_index = 0
        self._sample_i = None
        self._v1 = None
        self._h = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs,
    ):
        if self._step_index is None:
            self._step_index = 0

        idx = self._step_index
        is_predictor = (idx % 2 == 0)
        interval_idx = idx // 2

        s_curr = self.sigmas[2 * interval_idx]
        s_next = self.sigmas[2 * interval_idx + 1]
        dt = s_next - s_curr

        if is_predictor:
            self._sample_i = sample
            self._v1 = model_output
            self._h = dt
            prev_sample = sample + dt * model_output
        else:
            v1 = self._v1 if self._v1 is not None else model_output
            v2 = model_output
            sample_0 = self._sample_i if self._sample_i is not None else sample
            dt = self._h if self._h is not None else dt

            prev_sample = sample_0 + (dt / 2.0) * (v1 + v2)

            self._sample_i = None
            self._v1 = None
            self._h = None

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

SCHEDULER_REGISTRY = {
    "euler": FlowMatchEulerDiscreteScheduler,
    "heun": MiniMaxFlowMatchHeunDiscreteScheduler,
}

def apply_blue_noise_dispersion(tensor: torch.Tensor, alpha: float) -> torch.Tensor:
    orig_dtype = tensor.dtype
    work_tensor = tensor.to(dtype=torch.float32)
    n_fft = work_tensor.shape[-1]
    spectrum = torch.fft.rfft(work_tensor, n=n_fft, dim=-1)
    k = torch.arange(spectrum.shape[-1], device=tensor.device, dtype=torch.float32)
    norm_freq = k / float(max(spectrum.shape[-1] - 1, 1))

    spectral_filter = torch.pow(torch.clamp(norm_freq, min=1e-4), alpha)
    spectral_filter[0] = 0.0

    filtered_spectrum = spectrum * spectral_filter
    filtered_tensor = torch.fft.irfft(filtered_spectrum, n=n_fft, dim=-1)

    mean = torch.mean(filtered_tensor)
    std = torch.std(filtered_tensor)
    standardized = (filtered_tensor - mean) / torch.clamp(std, min=1e-8)
    return standardized.to(dtype=orig_dtype)

def apply_perona_malik_diffusion(
    tensor: torch.Tensor,
    iterations: int,
    conductance: float,
    stability_lambda: float
) -> torch.Tensor:
    orig_dtype = tensor.dtype
    u = tensor.to(dtype=torch.float32).clone()
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
    standardized = (u - mean) / torch.clamp(std, min=1e-8)
    return standardized.to(dtype=orig_dtype)

def apply_sub_millisecond_declick(audio_tensor: torch.Tensor, fade_samples: int = 128) -> torch.Tensor:
    if audio_tensor.shape[-1] <= fade_samples:
        return audio_tensor
    fade_curve = 0.5 * (1.0 - torch.cos(torch.linspace(0.0, math.pi, fade_samples, device=audio_tensor.device, dtype=audio_tensor.dtype)))
    audio_tensor[..., :fade_samples] *= fade_curve
    return audio_tensor

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

        self._pristine_scheduler_cls = None
        self._pristine_scheduler_config = {}
        self._pristine_guidance_scale = None
        self._pristine_gen_configs: Dict[int, Any] = {}

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

    def _snapshot_pristine_state(self) -> None:
        if hasattr(self.pipe, "scheduler") and self.pipe.scheduler is not None:
            self._pristine_scheduler_cls = self.pipe.scheduler.__class__
            self._pristine_scheduler_config = copy.deepcopy(dict(getattr(self.pipe.scheduler, "config", {})))

        if hasattr(self.pipe, "guider") and self.pipe.guider is not None:
            if hasattr(self.pipe.guider, "guidance_scale"):
                self._pristine_guidance_scale = self.pipe.guider.guidance_scale
            elif hasattr(self.pipe.guider, "config") and isinstance(self.pipe.guider.config, dict):
                self._pristine_guidance_scale = self.pipe.guider.config.get("guidance_scale", 1.0)

        for mod in self._get_ar_candidate_modules():
            if hasattr(mod, "generation_config") and mod.generation_config is not None:
                self._pristine_gen_configs[id(mod)] = copy.deepcopy(mod.generation_config)

    def _get_ar_candidate_modules(self) -> list:
        candidate_modules = []
        components = getattr(self.pipe, "components", {})
        if isinstance(components, dict):
            candidate_modules.extend(components.values())
        for attr in ["language_model", "text_encoder", "condition_encoder", "semantic_generator"]:
            if hasattr(self.pipe, attr):
                candidate_modules.append(getattr(self.pipe, attr))
        return [m for m in candidate_modules if m is not None]

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
        self._snapshot_pristine_state()

    def _configure_scheduler(
        self,
        scheduler_type: str,
        audio_duration: float,
        sampling_rate: int,
        upsampling_factor: int = 512
    ) -> None:
        if hasattr(self.pipe, "scheduler") and self.pipe.scheduler is not None:
            config = copy.deepcopy(self._pristine_scheduler_config)
            base_shift = config.get("base_shift", 0.5)
            max_shift = config.get("max_shift", 1.15)
            base_seq_len = config.get("base_image_seq_len", 256)
            max_seq_len = config.get("max_image_seq_len", 4096)

            latent_seq_len = int(math.ceil((audio_duration * sampling_rate) / upsampling_factor))
            ratio = max(0.0, min(1.0, (latent_seq_len - base_seq_len) / float(max_seq_len - base_seq_len)))
            dynamic_shift = base_shift + ratio * (max_shift - base_shift)

            if scheduler_type == "native" or scheduler_type not in SCHEDULER_REGISTRY:
                scheduler_cls = self._pristine_scheduler_cls
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
        target_scale = guidance_scale if guidance_scale is not None else self._pristine_guidance_scale
        if hasattr(self.pipe, "guider") and self.pipe.guider is not None and target_scale is not None:
            if hasattr(self.pipe.guider, "guidance_scale"):
                self.pipe.guider.guidance_scale = target_scale
            elif hasattr(self.pipe.guider, "config") and isinstance(self.pipe.guider.config, dict):
                self.pipe.guider.config["guidance_scale"] = target_scale

    def _configure_autoregressive_sampling(
        self,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int]
    ) -> None:
        for mod in self._get_ar_candidate_modules():
            if id(mod) in self._pristine_gen_configs:
                mod.generation_config = copy.deepcopy(self._pristine_gen_configs[id(mod)])
                if temperature is not None:
                    mod.generation_config.temperature = temperature
                    mod.generation_config.do_sample = True
                if top_p is not None:
                    mod.generation_config.top_p = top_p
                    mod.generation_config.do_sample = True
                if top_k is not None:
                    mod.generation_config.top_k = top_k
                    mod.generation_config.do_sample = True

    def _apply_latent_pre_hook(self, request: GenerationRequest):
        if request.noise_topology == "gaussian":
            return None

        target_transformer = getattr(self.pipe, "transformer", None)
        if target_transformer is None and hasattr(self.pipe, "components"):
            target_transformer = self.pipe.components.get("transformer", None)

        if target_transformer is None:
            return None

        is_first_evaluation = True

        def shape_latents(tensor: torch.Tensor) -> torch.Tensor:
            if request.noise_topology == "blue_noise":
                return apply_blue_noise_dispersion(tensor, alpha=request.blue_noise_alpha)
            elif request.noise_topology == "perona_malik":
                return apply_perona_malik_diffusion(
                    tensor,
                    iterations=request.pm_iterations,
                    conductance=request.pm_conductance,
                    stability_lambda=request.pm_lambda
                )
            return tensor

        def pre_hook(module, args, kwargs):
            nonlocal is_first_evaluation
            if is_first_evaluation:
                is_first_evaluation = False
                if len(args) > 0 and isinstance(args[0], torch.Tensor):
                    modified_latent = shape_latents(args[0])
                    new_args = (modified_latent,) + args[1:]
                    return new_args, kwargs
                elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                    kwargs["hidden_states"] = shape_latents(kwargs["hidden_states"])
                    return args, kwargs
            return args, kwargs

        try:
            return target_transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)
        except TypeError:
            def legacy_hook(module, args):
                nonlocal is_first_evaluation
                if is_first_evaluation:
                    is_first_evaluation = False
                    if len(args) > 0 and isinstance(args[0], torch.Tensor):
                        modified_latent = shape_latents(args[0])
                        return (modified_latent,) + args[1:]
                return args
            return target_transformer.register_forward_pre_hook(legacy_hook)

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

        pipeline_kwargs = {
            "prompt": effective_prompt,
            "lyrics": sanitized_lyrics,
            "audio_duration": request.audio_duration,
            "generator": generator,
            "output": "audios",
        }

        if request.num_inference_steps is not None:
            pipeline_kwargs["num_inference_steps"] = request.num_inference_steps

        hook_handle = self._apply_latent_pre_hook(request)
        start_time = time.perf_counter()

        try:
            with torch.inference_mode():
                raw_output = self.pipe(**pipeline_kwargs)[0]
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        elapsed_time = time.perf_counter() - start_time

        if isinstance(raw_output, torch.Tensor):
            audio_tensor = raw_output.to(device=self.device, dtype=torch.float32)
        else:
            audio_tensor = torch.as_tensor(raw_output, device=self.device, dtype=torch.float32)

        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.ndim == 3:
            audio_tensor = audio_tensor.squeeze(0)

        audio_tensor = audio_tensor - torch.mean(audio_tensor, dim=-1, keepdim=True)
        audio_tensor = apply_sub_millisecond_declick(audio_tensor, fade_samples=128)

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