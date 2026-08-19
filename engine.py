# engine.py
import os
import sys
import gc
import math
import copy
import time
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
warnings.filterwarnings("ignore", message=".*Modular Diffusers is currently an experimental feature.*")
warnings.filterwarnings("ignore", message=".*Guiders are currently an experimental feature.*")
warnings.filterwarnings("ignore", message=".*Casting directly with `to()` can lead to inconsistent results.*")

import numpy as np
import soundfile as sf
import torch
import torch.fft
import torch.nn.functional as F

from diffusers import (
    ModularPipeline,
    ComponentsManager,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput

from schema import GenerationRequest, GenerationResponse


def extract_timestep_float(t_val: Any) -> Optional[float]:
    if t_val is None:
        return None
    if isinstance(t_val, (int, float)):
        return float(t_val)
    if isinstance(t_val, torch.Tensor):
        if t_val.numel() == 0:
            return None
        return float(t_val.detach().reshape(-1)[0].item())
    return None


class MiniMaxFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_index: Optional[int] = None

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
        self._step_index = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs,
    ):
        num_steps = len(self.timesteps)
        if self._step_index is None or self._step_index >= num_steps:
            self._step_index = 0

        idx = self._step_index
        s_curr = self.sigmas[idx]
        s_next = self.sigmas[idx + 1]
        dt = s_next - s_curr

        prev_sample = sample + dt * model_output

        self._step_index += 1
        if self._step_index >= num_steps:
            self._step_index = None

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)


class MiniMaxFlowMatchHeunDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_i: Optional[torch.Tensor] = None
        self._v1: Optional[torch.Tensor] = None
        self._h: Optional[torch.Tensor] = None
        self._step_index: Optional[int] = None

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

        target_device = device if device is not None else (
            base_timesteps.device if isinstance(base_timesteps, torch.Tensor) else "cpu"
        )
        base_sigmas = base_sigmas.to(device=target_device)
        base_timesteps = base_timesteps.to(device=target_device)

        num_intervals = len(base_sigmas) - 1
        if num_intervals <= 0:
            return

        denom = base_sigmas[0] if base_sigmas[0].abs() > 1e-8 else torch.tensor(1.0, device=target_device)
        scale = base_timesteps[0] / denom
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
            else torch.tensor(heun_timesteps, device=target_device)
        )
        self.sigmas = (
            torch.stack(heun_sigmas)
            if isinstance(heun_sigmas[0], torch.Tensor)
            else torch.tensor(heun_sigmas, device=target_device)
        )

        self._step_index = None
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
        num_steps = len(self.timesteps)
        if self._step_index is None or self._step_index >= num_steps:
            self._step_index = 0
            self._sample_i = None
            self._v1 = None
            self._h = None

        idx = self._step_index
        is_predictor = (idx % 2 == 0)
        interval_idx = idx // 2

        s_curr = self.sigmas[2 * interval_idx]
        s_next = self.sigmas[2 * interval_idx + 1]
        dt = s_next - s_curr

        if is_predictor:
            self._sample_i = sample.clone()
            self._v1 = model_output.clone()
            self._h = dt
            prev_sample = sample + dt * model_output
        else:
            v1 = self._v1 if self._v1 is not None else model_output
            sample_0 = self._sample_i if self._sample_i is not None else sample
            dt = self._h if self._h is not None else dt

            prev_sample = sample_0 + (dt / 2.0) * (v1 + model_output)
            self._sample_i = None
            self._v1 = None
            self._h = None

        self._step_index += 1
        if self._step_index >= num_steps:
            self._step_index = None
            self._sample_i = None
            self._v1 = None
            self._h = None

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)


SCHEDULER_REGISTRY = {
    "native": FlowMatchEulerDiscreteScheduler,
    "euler": MiniMaxFlowMatchEulerDiscreteScheduler,
    "heun": MiniMaxFlowMatchHeunDiscreteScheduler,
}


def apply_dct_2(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    N = x.shape[dim]
    if dim != -1 and dim != x.ndim - 1:
        x = x.transpose(dim, -1)
    orig_shape = x.shape
    x_2d = x.reshape(-1, N)

    idx = torch.empty(N, dtype=torch.long, device=x.device)
    idx[:(N + 1) // 2] = torch.arange(0, N, 2, device=x.device)
    idx[(N + 1) // 2:] = torch.arange(N - 1 - (N % 2), 0, -2, device=x.device)

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
    gamma_base = math.sqrt(float(N) / float(torch.sum(H_k ** 2).clamp(min=1e-8).item()))

    theta = (math.pi / 2.0) * min(max(blend_homotopy, 0.0), 1.0)
    G_k = math.cos(theta) + math.sin(theta) * (gamma_base * H_k)
    gamma_theta = math.sqrt(float(N) / float(torch.sum(G_k ** 2).clamp(min=1e-8).item()))
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
        k_eff = conductance * math.sqrt(1.0 + 2.0 * (blue_noise_alpha ** 2))
    else:
        k_eff = conductance
    k_sq = max(k_eff ** 2, 1e-8)

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

        c_east = torch.exp(-(grad_east ** 2) / k_sq)
        c_west = torch.exp(-(grad_west ** 2) / k_sq)

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
    fade = 0.5 * (1.0 - torch.cos(torch.linspace(0.0, math.pi, fade_samples, device=audio_tensor.device, dtype=audio_tensor.dtype)))
    audio_tensor[..., :fade_samples] *= fade
    audio_tensor[..., -fade_samples:] *= torch.flip(fade, dims=[0])
    return audio_tensor


class MusicEngine:
    def __init__(
        self,
        repo_id: str = "MiniMaxAI/MiniMax-Music3",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        self.repo_id = repo_id
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.pipe: Optional[ModularPipeline] = None
        self._current_offload_state: Optional[bool] = None

        self._pristine_scheduler_cls = None
        self._pristine_scheduler_config = {}

    def _fold_weight_norm(self, module: torch.nn.Module) -> None:
        for sub_module in module.modules():
            try:
                torch.nn.utils.remove_weight_norm(sub_module)
            except (ValueError, AttributeError):
                pass

    def _get_module(self, name: str) -> Optional[torch.nn.Module]:
        mod = getattr(self.pipe, name, None)
        if mod is None and hasattr(self.pipe, "components") and isinstance(self.pipe.components, dict):
            mod = self.pipe.components.get(name, None)
        return mod if isinstance(mod, torch.nn.Module) else None

    @staticmethod
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

    def _set_eval(self) -> None:
        for attr in dir(self.pipe):
            if not attr.startswith("_"):
                try:
                    val = getattr(self.pipe, attr)
                    if isinstance(val, torch.nn.Module):
                        val.eval()
                except Exception:
                    pass

    def _snapshot_pristine_state(self) -> None:
        if hasattr(self.pipe, "scheduler") and self.pipe.scheduler is not None:
            self._pristine_scheduler_cls = self.pipe.scheduler.__class__
            self._pristine_scheduler_config = copy.deepcopy(dict(getattr(self.pipe.scheduler, "config", {})))

    def _ensure_pipeline(self, cpu_offload: bool) -> None:
        if self.pipe is not None and self._current_offload_state == cpu_offload:
            return

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP execution requested but no compatible device was detected.")

        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        kwargs = {"local_files_only": True}
        if cpu_offload:
            mgr = ComponentsManager()
            mgr.enable_auto_cpu_offload(device=str(self.device))
            kwargs["components_manager"] = mgr

        self.pipe = ModularPipeline.from_pretrained(self.repo_id, **kwargs)
        self.pipe.load_components(dtype=self.dtype, local_files_only=True)

        rvq_mod = self._get_module("rvq_depth_decoder")
        if rvq_mod is not None:
            self._fold_weight_norm(rvq_mod)
            rvq_mod.to(dtype=self.dtype)

        lm_mod = self._get_module("language_model")
        if lm_mod is not None:
            lm_mod.to(dtype=self.dtype)

        transformer_mod = self._get_module("transformer")
        if transformer_mod is not None:
            transformer_mod.to(dtype=self.dtype)

        vocoder_mod = self._get_module("vocoder")
        if vocoder_mod is not None:
            self._fold_weight_norm(vocoder_mod)
            vocoder_mod.to(dtype=torch.float32)
            try:
                vocoder_mod.register_forward_pre_hook(self._cast_inputs_to_fp32, with_kwargs=True)
            except TypeError:
                vocoder_mod.register_forward_pre_hook(
                    lambda m, a: tuple(x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in a)
                )

        audio_vae_mod = self._get_module("audio_vae")
        if audio_vae_mod is not None:
            self._fold_weight_norm(audio_vae_mod)
            audio_vae_mod.to(dtype=torch.float32)
            try:
                audio_vae_mod.register_forward_pre_hook(self._cast_inputs_to_fp32, with_kwargs=True)
            except TypeError:
                audio_vae_mod.register_forward_pre_hook(
                    lambda m, a: tuple(x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in a)
                )

        if not cpu_offload:
            self.pipe.to(self.device)
            if rvq_mod is not None:
                rvq_mod.to(device=self.device, dtype=self.dtype)
            if lm_mod is not None:
                lm_mod.to(device=self.device, dtype=self.dtype)
            if transformer_mod is not None:
                transformer_mod.to(device=self.device, dtype=self.dtype)
            if vocoder_mod is not None:
                vocoder_mod.to(device=self.device, dtype=torch.float32)
            if audio_vae_mod is not None:
                audio_vae_mod.to(device=self.device, dtype=torch.float32)

        self._set_eval()
        self._snapshot_pristine_state()
        self._current_offload_state = cpu_offload

    def _configure_scheduler(
        self,
        scheduler_type: str,
        audio_duration: float,
        sampling_rate: int = 32000,
        upsampling_factor: int = 512
    ) -> None:
        if hasattr(self.pipe, "scheduler") and self.pipe.scheduler is not None:
            cfg = copy.deepcopy(self._pristine_scheduler_config)
            cls_ = SCHEDULER_REGISTRY.get(scheduler_type, FlowMatchEulerDiscreteScheduler)

            base_shift = cfg.get("base_shift", 0.5)
            max_shift = cfg.get("max_shift", 1.15)
            base_seq_len = cfg.get("base_image_seq_len", 256)
            max_seq_len = cfg.get("max_image_seq_len", 4096)

            latent_seq_len = int(math.ceil((audio_duration * sampling_rate) / upsampling_factor))
            ratio = max(0.0, min(1.0, (latent_seq_len - base_seq_len) / float(max_seq_len - base_seq_len)))
            dynamic_shift = base_shift + ratio * (max_shift - base_shift)

            new_sched = cls_.from_config(
                cfg,
                shift=dynamic_shift,
                use_dynamic_shifting=False
            )
            self.pipe.scheduler = new_sched
            if hasattr(self.pipe, "components") and isinstance(self.pipe.components, dict):
                self.pipe.components["scheduler"] = new_sched

    def _configure_guider(self, guidance_scale: Optional[float]) -> None:
        if guidance_scale is None:
            return

        target_scale = float(guidance_scale)
        guider_objs = []

        if hasattr(self.pipe, "guider") and getattr(self.pipe, "guider") is not None:
            guider_objs.append(getattr(self.pipe, "guider"))

        if hasattr(self.pipe, "components") and isinstance(self.pipe.components, dict):
            if "guider" in self.pipe.components and self.pipe.components["guider"] is not None:
                guider_objs.append(self.pipe.components["guider"])

        blocks_target = getattr(self.pipe, "blocks", None)
        if blocks_target is None and hasattr(self.pipe, "current_pipeline"):
            blocks_target = getattr(self.pipe.current_pipeline, "blocks", None)

        if blocks_target is not None:
            items = blocks_target.items() if isinstance(blocks_target, dict) else (
                enumerate(blocks_target) if isinstance(blocks_target, (list, tuple)) else []
            )
            for _, b in items:
                if hasattr(b, "guider") and getattr(b, "guider") is not None:
                    guider_objs.append(getattr(b, "guider"))
                if hasattr(b, "guidance_scale"):
                    try:
                        setattr(b, "guidance_scale", target_scale)
                    except Exception:
                        pass

        for g in guider_objs:
            if hasattr(g, "guidance_scale"):
                try:
                    g.guidance_scale = target_scale
                except Exception:
                    pass
            if hasattr(g, "config") and isinstance(g.config, dict) and "guidance_scale" in g.config:
                g.config["guidance_scale"] = target_scale

    def _apply_global_latent_shaping(self, request: GenerationRequest):
        shaping_applied = [False]

        def shape_latents(tensor: torch.Tensor) -> torch.Tensor:
            out = tensor
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

        restorations = []
        target = getattr(self.pipe, "transformer", None)
        if target is None and hasattr(self.pipe, "components") and isinstance(self.pipe.components, dict):
            target = self.pipe.components.get("transformer", None)

        if target is not None:
            def transformer_pre_hook(module, args, kwargs):
                target_tensor = args[0] if len(args) > 0 and isinstance(args[0], torch.Tensor) else kwargs.get("hidden_states")
                if target_tensor is None:
                    return args, kwargs

                raw_t = kwargs.get("timestep", args[1] if len(args) > 1 else None)
                t_curr = extract_timestep_float(raw_t)

                sched = getattr(self.pipe, "scheduler", None)
                t_init = None
                if sched is not None and hasattr(sched, "timesteps") and sched.timesteps is not None and len(sched.timesteps) > 0:
                    t_init = extract_timestep_float(sched.timesteps[0])

                is_step0 = False
                if t_curr is not None and t_init is not None:
                    is_step0 = bool(abs(t_curr - t_init) < 1e-3)

                if is_step0 and not shaping_applied[0]:
                    with torch.no_grad():
                        target_tensor.copy_(shape_latents(target_tensor))
                    shaping_applied[0] = True

                return args, kwargs

            try:
                handle = target.register_forward_pre_hook(transformer_pre_hook, with_kwargs=True)
            except TypeError:
                def simple_hook(m, a):
                    if len(a) > 0 and isinstance(a[0], torch.Tensor):
                        tens = a[0]
                        if not shaping_applied[0]:
                            tens = shape_latents(tens)
                            shaping_applied[0] = True
                        return (tens, *a[1:])
                    return a
                handle = target.register_forward_pre_hook(simple_hook)
            restorations.append(("hook_handle", handle))

        return restorations

    def _apply_stage1_generation_hook(self, request: GenerationRequest, duration: float):
        lm_target = self._get_module("language_model")
        if lm_target is None or not hasattr(lm_target, "generate"):
            return None

        orig_generate = lm_target.generate
        min_needed_tokens = max(512, int(math.floor(duration * 24.5)))
        needed_tokens = max(2048, int(math.ceil(duration * 35)) + 512)
        extended_max_length = needed_tokens + 4096

        gen_cfg = getattr(lm_target, "generation_config", None)
        orig_cfg_state = {}
        if gen_cfg is not None:
            for param in [
                "temperature", "top_p", "top_k", "do_sample",
                "max_new_tokens", "min_new_tokens", "max_length"
            ]:
                if hasattr(gen_cfg, param):
                    orig_cfg_state[param] = getattr(gen_cfg, param)

            if request.temperature is not None:
                gen_cfg.temperature = float(request.temperature)
                gen_cfg.do_sample = True
            if request.top_p is not None:
                gen_cfg.top_p = float(request.top_p)
                gen_cfg.do_sample = True
            if request.top_k is not None:
                gen_cfg.top_k = int(request.top_k)
                gen_cfg.do_sample = True

            gen_cfg.max_new_tokens = needed_tokens
            gen_cfg.min_new_tokens = min_needed_tokens
            gen_cfg.max_length = extended_max_length

        blocks_target = getattr(self.pipe, "blocks", None)
        if blocks_target is None and hasattr(self.pipe, "current_pipeline"):
            blocks_target = getattr(self.pipe.current_pipeline, "blocks", None)

        orig_block_params = []
        if blocks_target is not None:
            items = blocks_target.items() if isinstance(blocks_target, dict) else (
                enumerate(blocks_target) if isinstance(blocks_target, (list, tuple)) else []
            )
            for _, b in items:
                for param, val in [
                    ("temperature", request.temperature),
                    ("top_p", request.top_p),
                    ("top_k", request.top_k),
                    ("max_new_tokens", needed_tokens),
                    ("min_new_tokens", min_needed_tokens),
                    ("max_length", extended_max_length),
                ]:
                    if val is not None and hasattr(b, param):
                        orig_val = getattr(b, param)
                        orig_block_params.append((b, param, orig_val))
                        try:
                            setattr(b, param, val)
                        except Exception:
                            pass

        def generation_wrapper(*args, **kwargs):
            if request.temperature is not None:
                kwargs["temperature"] = float(request.temperature)
                kwargs["do_sample"] = True
            if request.top_p is not None:
                kwargs["top_p"] = float(request.top_p)
                kwargs["do_sample"] = True
            if request.top_k is not None:
                kwargs["top_k"] = int(request.top_k)
                kwargs["do_sample"] = True

            kwargs["max_new_tokens"] = needed_tokens
            kwargs["min_new_tokens"] = min_needed_tokens
            kwargs["max_length"] = extended_max_length

            if request.enable_speculative_markov:
                kwargs["prompt_lookup_num_tokens"] = request.speculative_draft_k
                if request.temperature is None and "do_sample" not in kwargs:
                    kwargs["do_sample"] = False
            else:
                kwargs.pop("prompt_lookup_num_tokens", None)

            return orig_generate(*args, **kwargs)

        lm_target.generate = generation_wrapper
        return (lm_target, orig_generate, gen_cfg, orig_cfg_state, orig_block_params)

    def synthesize(self, request: GenerationRequest) -> GenerationResponse:
        request.validate()
        self._ensure_pipeline(request.cpu_offload)

        effective_prompt = request.compile_prompt()
        sanitized_lyrics = request.sanitize_lyrics()

        sampling_rate = getattr(self.pipe, "sampling_rate", None)
        if sampling_rate is None and hasattr(self.pipe, "vocoder"):
            sampling_rate = getattr(self.pipe.vocoder, "config", {}).get("sampling_rate", 32000)
        if sampling_rate is None:
            sampling_rate = 32000

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.perf_counter()

        self._configure_scheduler(
            scheduler_type=request.scheduler_type,
            audio_duration=request.audio_duration,
            sampling_rate=sampling_rate,
            upsampling_factor=512
        )

        if request.seed is not None and request.seed >= 0:
            torch.manual_seed(request.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request.seed)
            np.random.seed(request.seed % (2**32))
            generator = torch.Generator(device=self.device).manual_seed(request.seed)
        else:
            generator = None

        pipeline_kwargs = {
            "prompt": effective_prompt,
            "lyrics": sanitized_lyrics,
            "audio_duration": request.audio_duration,
            "generator": generator,
            "output": "audios",
        }

        if request.num_inference_steps is not None:
            pipeline_kwargs["num_inference_steps"] = request.num_inference_steps

        self._configure_guider(request.guidance_scale)
        restorations = self._apply_global_latent_shaping(request=request)
        lm_hook_data = self._apply_stage1_generation_hook(
            request=request,
            duration=request.audio_duration
        )

        try:
            with torch.inference_mode():
                raw_output = self.pipe(**pipeline_kwargs)[0]
        finally:
            if restorations:
                for item in restorations:
                    if isinstance(item, tuple) and len(item) == 2:
                        target_obj, orig_fn = item
                        if target_obj == "hook_handle":
                            orig_fn.remove()
                        else:
                            target_obj.__call__ = orig_fn

            if lm_hook_data is not None:
                target_lm, orig_gen, gen_cfg, orig_cfg_state, orig_block_params = lm_hook_data
                target_lm.generate = orig_gen
                if gen_cfg is not None and orig_cfg_state:
                    for k, v in orig_cfg_state.items():
                        try:
                            setattr(gen_cfg, k, v)
                        except Exception:
                            pass
                for b, param, orig_val in orig_block_params:
                    try:
                        setattr(b, param, orig_val)
                    except Exception:
                        pass

        elapsed_time = time.perf_counter() - start_time

        peak_vram_gb = 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

        if isinstance(raw_output, torch.Tensor):
            audio_tensor = raw_output.to(device=self.device, dtype=torch.float32)
        else:
            audio_tensor = torch.as_tensor(raw_output, device=self.device, dtype=torch.float32)

        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.ndim == 3:
            audio_tensor = audio_tensor.squeeze(0)

        if request.apply_declick:
            audio_tensor = apply_sub_millisecond_declick(audio_tensor, fade_samples=512)

        peak_val = torch.max(torch.abs(audio_tensor)).item()
        rms_val = torch.sqrt(torch.mean(audio_tensor ** 2)).item()
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
            speculative_markov_used=request.enable_speculative_markov,
            noise_topology_used=request.noise_topology,
            pm_diffusion_used=request.enable_pm_diffusion,
            effective_prompt=effective_prompt,
            declick_applied=request.apply_declick,
            cpu_offload_active=bool(self._current_offload_state),
            peak_vram_gb=peak_vram_gb
        )