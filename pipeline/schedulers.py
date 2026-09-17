import math
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np
import torch

class FlowMatchEulerDiscreteScheduler:
    def __init__(self, shift: float = 1.0, num_train_timesteps: int = 1):
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps
        self.timesteps: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self._step_index: Optional[int] = None

    def reset(self) -> None:
        self._step_index = 0

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[str, torch.device]] = None,
        shift: Optional[float] = None,
    ) -> None:
        if shift is not None:
            self.shift = shift
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=np.float64)
        if self.shift != 1.0:
            sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        sigmas_tensor = 1.0 - torch.from_numpy(sigmas.astype(np.float32)).to(device=target_device)
        self.timesteps = sigmas_tensor * self.num_train_timesteps
        self.sigmas = torch.cat([sigmas_tensor, torch.ones(1, device=target_device, dtype=torch.float32)])
        self._step_index = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        num_steps = len(self.timesteps)
        if self._step_index is None or self._step_index >= num_steps:
            self._step_index = 0
        idx = self._step_index
        sigma_curr = self.sigmas[idx]
        sigma_next = self.sigmas[idx + 1]
        dt = sigma_next - sigma_curr
        prev_sample = sample + dt * model_output
        self._step_index += 1
        if self._step_index >= num_steps:
            self._step_index = None
        return prev_sample

class FlowMatchHeunDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, shift: float = 1.0, num_train_timesteps: int = 1):
        super().__init__(shift=shift, num_train_timesteps=num_train_timesteps)
        self._sample_i: Optional[torch.Tensor] = None
        self._v1: Optional[torch.Tensor] = None
        self._h: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._step_index = 0
        self._sample_i = None
        self._v1 = None
        self._h = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[str, torch.device]] = None,
        shift: Optional[float] = None,
    ) -> None:
        super().set_timesteps(num_inference_steps=num_inference_steps, device=device, shift=shift)
        base_sigmas = self.sigmas
        target_device = device if device is not None else base_sigmas.device
        num_intervals = len(base_sigmas) - 1
        if num_intervals <= 0:
            return
        heun_sigmas = []
        for i in range(num_intervals):
            s_curr = base_sigmas[i]
            s_next = base_sigmas[i + 1]
            heun_sigmas.extend([s_curr, s_next])
        heun_sigmas.append(base_sigmas[-1])
        self.sigmas = torch.stack(heun_sigmas).to(device=target_device)
        self.timesteps = (self.sigmas[:-1] * self.num_train_timesteps).clone()
        self._step_index = None
        self._sample_i = None
        self._v1 = None
        self._h = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
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
        return prev_sample

class FlowMatchIPNDMDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(
        self,
        shift: float = 1.0,
        num_train_timesteps: int = 1,
        max_order: int = 4,
    ):
        super().__init__(shift=shift, num_train_timesteps=num_train_timesteps)
        self.max_order = max(1, min(4, int(max_order)))
        self.buffer_derivatives: List[torch.Tensor] = []
        self._sigmas_cpu: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._step_index = 0
        self.buffer_derivatives.clear()

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[str, torch.device]] = None,
        shift: Optional[float] = None,
    ) -> None:
        super().set_timesteps(num_inference_steps=num_inference_steps, device=device, shift=shift)
        self.buffer_derivatives.clear()
        self._sigmas_cpu = self.sigmas.detach().cpu().numpy()

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        num_steps = len(self.timesteps)
        if self._step_index is None or self._step_index >= num_steps:
            self._step_index = 0
            self.buffer_derivatives.clear()
            if self._sigmas_cpu is None:
                self._sigmas_cpu = self.sigmas.detach().cpu().numpy()
        idx = self._step_index
        sigma_curr = self.sigmas[idx]
        sigma_next = self.sigmas[idx + 1]
        dt = sigma_next - sigma_curr
        d_cur = model_output
        order = min(self.max_order, idx + 1)
        if order == 1 or len(self.buffer_derivatives) == 0:
            dx = dt * d_cur
        elif order == 2 or len(self.buffer_derivatives) == 1:
            h_n = float(self._sigmas_cpu[idx + 1] - self._sigmas_cpu[idx])
            h_n_1 = float(self._sigmas_cpu[idx] - self._sigmas_cpu[idx - 1])
            r = h_n / max(h_n_1, 1e-12)
            c1 = (2.0 + r) / 2.0
            c2 = -r / 2.0
            dx = dt * (c1 * d_cur + c2 * self.buffer_derivatives[-1])
        elif order == 3 or len(self.buffer_derivatives) == 2:
            h_n = float(self._sigmas_cpu[idx + 1] - self._sigmas_cpu[idx])
            h_n_1 = float(self._sigmas_cpu[idx] - self._sigmas_cpu[idx - 1])
            h_n_2 = float(self._sigmas_cpu[idx - 1] - self._sigmas_cpu[idx - 2])
            temp = (
                1.0
                - (h_n / (3.0 * max(h_n + h_n_1, 1e-12)))
                * ((h_n * (h_n + h_n_1)) / max(h_n_1 * (h_n_1 + h_n_2), 1e-12))
            ) / 2.0
            c1 = (2.0 + (h_n / max(h_n_1, 1e-12))) / 2.0 + temp
            c2 = -(h_n / max(h_n_1, 1e-12)) / 2.0 - (1.0 + h_n_1 / max(h_n_2, 1e-12)) * temp
            c3 = temp * (h_n_1 / max(h_n_2, 1e-12))
            dx = dt * (
                c1 * d_cur
                + c2 * self.buffer_derivatives[-1]
                + c3 * self.buffer_derivatives[-2]
            )
        else:
            h_n = float(self._sigmas_cpu[idx + 1] - self._sigmas_cpu[idx])
            h_n_1 = float(self._sigmas_cpu[idx] - self._sigmas_cpu[idx - 1])
            h_n_2 = float(self._sigmas_cpu[idx - 1] - self._sigmas_cpu[idx - 2])
            h_n_3 = float(self._sigmas_cpu[idx - 2] - self._sigmas_cpu[idx - 3])
            temp1 = (
                1.0
                - (h_n / (3.0 * max(h_n + h_n_1, 1e-12)))
                * ((h_n * (h_n + h_n_1)) / max(h_n_1 * (h_n_1 + h_n_2), 1e-12))
            ) / 2.0
            temp2 = (
                (1.0 - h_n / (3.0 * max(h_n + h_n_1, 1e-12))) / 2.0
                + (1.0 - h_n / (2.0 * max(h_n + h_n_1, 1e-12)))
                * h_n
                / (6.0 * max(h_n + h_n_1 + h_n_2, 1e-12))
            ) * (
                (h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2))
                / max(h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3), 1e-12)
            )
            c1 = (2.0 + (h_n / max(h_n_1, 1e-12))) / 2.0 + temp1 + temp2
            c2 = (
                -(h_n / max(h_n_1, 1e-12)) / 2.0
                - (1.0 + h_n_1 / max(h_n_2, 1e-12)) * temp1
                - (
                    1.0
                    + (h_n_1 / max(h_n_2, 1e-12))
                    + (h_n_1 * (h_n_1 + h_n_2)) / max(h_n_2 * (h_n_2 + h_n_3), 1e-12)
                )
                * temp2
            )
            c3 = (
                temp1 * (h_n_1 / max(h_n_2, 1e-12))
                + (
                    (h_n_1 / max(h_n_2, 1e-12))
                    + (h_n_1 * (h_n_1 + h_n_2)) / max(h_n_2 * (h_n_2 + h_n_3), 1e-12)
                )
                * (1.0 + h_n_2 / max(h_n_3, 1e-12))
                * temp2
            )
            c4 = (
                -temp2
                * ((h_n_1 * (h_n_1 + h_n_2)) / max(h_n_2 * (h_n_2 + h_n_3), 1e-12))
                * (h_n_1 / max(h_n_2, 1e-12))
            )
            dx = dt * (
                c1 * d_cur
                + c2 * self.buffer_derivatives[-1]
                + c3 * self.buffer_derivatives[-2]
                + c4 * self.buffer_derivatives[-3]
            )
        if len(self.buffer_derivatives) == self.max_order - 1:
            self.buffer_derivatives.pop(0)
        self.buffer_derivatives.append(d_cur.detach())
        prev_sample = sample + dx
        self._step_index += 1
        if self._step_index >= num_steps:
            self._step_index = None
            self.buffer_derivatives.clear()
        return prev_sample

class BifurcatedFlowMatchScheduler:
    def __init__(
        self,
        early_solver: str = "heun",
        late_solver: str = "ipndm",
        handoff_threshold: float = 0.3257,
        shift: float = 1.0,
        num_train_timesteps: int = 1,
        ipndm_max_order: int = 4,
        eta: float = 0.0,
        s_noise: float = 1.0,
        **kwargs: Any,
    ):
        early_cand = kwargs.get("early_instrumental_solver") or kwargs.get("early_vocal_solver") or early_solver
        late_cand = kwargs.get("late_instrumental_solver") or kwargs.get("late_vocal_solver") or late_solver
        self.early_solver = str(early_cand).lower()
        self.late_solver = str(late_cand).lower()
        self.handoff_threshold = max(0.0, min(1.0, float(handoff_threshold)))
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps
        self.ipndm_max_order = max(1, min(4, int(ipndm_max_order)))
        self.eta = max(0.0, min(1.0, float(eta)))
        self.s_noise = max(0.0, float(s_noise))
        self.base_sigmas: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self._sigmas_cpu: Optional[np.ndarray] = None
        self._step_configs: List[Dict[str, Any]] = []
        self._step_index: Optional[int] = None
        self._k_star: int = 0
        self._sample_anchor: Optional[torch.Tensor] = None
        self._v0: Optional[torch.Tensor] = None
        self._h: Optional[float] = None
        self._history: List[torch.Tensor] = []
        self._boundary_flushed: bool = False
        self.has_corrector: bool = True
        self._early_steps: int = 0
        self._late_steps: int = 0

    def reset(self) -> None:
        self._step_index = 0
        self._sample_anchor = None
        self._v0 = None
        self._h = None
        self._history.clear()
        self._boundary_flushed = False

    @property
    def current_is_early(self) -> bool:
        if self._step_index is None or self._step_index >= len(self._step_configs):
            return True
        return bool(self._step_configs[self._step_index]["is_early"])

    def is_early_pass(self, pass_index: int) -> bool:
        if 0 <= pass_index < len(self._step_configs):
            return bool(self._step_configs[pass_index]["is_early"])
        return True

    @staticmethod
    def _compute_ab_weights(
        h_n: float,
        h_n_1: float,
        h_n_2: float,
        h_n_3: float,
        order: int,
    ) -> Tuple[float, float, float, float]:
        if order <= 1:
            return (h_n, 0.0, 0.0, 0.0)
        i0 = h_n
        i1 = 0.5 * (h_n**2)
        d1 = max(h_n_1, 1e-12)
        if order == 2:
            c0 = i0 + i1 / d1
            c1 = -i1 / d1
            return (c0, c1, 0.0, 0.0)
        i2 = (1.0 / 3.0) * (h_n**3) + 0.5 * d1 * (h_n**2)
        d2 = max(h_n_2, 1e-12)
        denom2 = d1 + d2
        alpha0 = 1.0 / (d1 * denom2)
        alpha1 = -1.0 / (d1 * d2)
        alpha2 = 1.0 / (d2 * denom2)
        if order == 3:
            c0 = i0 + (i1 / d1) + i2 * alpha0
            c1 = -(i1 / d1) + i2 * alpha1
            c2 = i2 * alpha2
            return (c0, c1, c2, 0.0)
        i3 = (
            0.25 * (h_n**4)
            + (1.0 / 3.0) * (2.0 * d1 + d2) * (h_n**3)
            + 0.5 * d1 * (d1 + d2) * (h_n**2)
        )
        d3 = max(h_n_3, 1e-12)
        denom3 = d1 + d2 + d3
        denom2_prev = d2 + d3
        beta1 = 1.0 / (d2 * denom2_prev)
        beta2 = -1.0 / (d2 * d3)
        beta3 = 1.0 / (d3 * denom2_prev)
        gamma0 = alpha0 / denom3
        gamma1 = (alpha1 - beta1) / denom3
        gamma2 = (alpha2 - beta2) / denom3
        gamma3 = -beta3 / denom3
        c0 = i0 + (i1 / d1) + i2 * alpha0 + i3 * gamma0
        c1 = -(i1 / d1) + i2 * alpha1 + i3 * gamma1
        c2 = i2 * alpha2 + i3 * gamma2
        c3 = i3 * gamma3
        return (c0, c1, c2, c3)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[str, torch.device]] = None,
        shift: Optional[float] = None,
        handoff_threshold: Optional[float] = None,
    ) -> None:
        if shift is not None:
            self.shift = shift
        if handoff_threshold is not None:
            self.handoff_threshold = max(0.0, min(1.0, float(handoff_threshold)))
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=np.float64)
        if self.shift != 1.0:
            sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)
        sigmas = 1.0 - sigmas
        base_sigmas_np = np.append(sigmas, 1.0)
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        self.base_sigmas = torch.from_numpy(base_sigmas_np.astype(np.float32)).to(device=target_device)
        self._sigmas_cpu = base_sigmas_np

        if self.handoff_threshold >= 1.0:
            k_star = num_inference_steps
        elif self.handoff_threshold <= 0.0:
            k_star = 0
        else:
            cand = [i for i in range(num_inference_steps) if base_sigmas_np[i] >= self.handoff_threshold]
            k_star = cand[0] if cand else num_inference_steps
        self._k_star = k_star
        self._early_steps = k_star
        self._late_steps = num_inference_steps - k_star

        corrector_types = ("heun", "sde_gpu_pp", "multitree", "res_multistep", "multires")
        early_has_corrector = self.early_solver in corrector_types
        late_has_corrector = self.late_solver in corrector_types

        step_configs: List[Dict[str, Any]] = []
        eval_timesteps: List[float] = []
        for i in range(num_inference_steps):
            is_early = (i < k_star)
            solver = self.early_solver if is_early else self.late_solver
            interval_corrector = early_has_corrector if is_early else late_has_corrector
            s_curr = float(base_sigmas_np[i])
            s_next = float(base_sigmas_np[i + 1])
            h_n = s_next - s_curr
            if interval_corrector:
                t_pred = s_curr * self.num_train_timesteps
                t_corr = s_next * self.num_train_timesteps
                eval_timesteps.extend([t_pred, t_corr])
                step_configs.append({
                    "mode": "predictor",
                    "interval_idx": i,
                    "is_early": is_early,
                    "s_curr": s_curr,
                    "s_next": s_next,
                    "dt": h_n,
                    "timestep": t_pred,
                    "solver": solver,
                })
                step_configs.append({
                    "mode": "corrector",
                    "interval_idx": i,
                    "is_early": is_early,
                    "s_curr": s_curr,
                    "s_next": s_next,
                    "dt": h_n,
                    "timestep": t_corr,
                    "solver": solver,
                })
            else:
                t_step = s_curr * self.num_train_timesteps
                eval_timesteps.append(t_step)
                if is_early:
                    order = min(self.ipndm_max_order, i + 1)
                    h_n_1 = float(base_sigmas_np[i] - base_sigmas_np[i - 1]) if i >= 1 else 0.0
                    h_n_2 = float(base_sigmas_np[i - 1] - base_sigmas_np[i - 2]) if i >= 2 else 0.0
                    h_n_3 = float(base_sigmas_np[i - 2] - base_sigmas_np[i - 3]) if i >= 3 else 0.0
                else:
                    j = i - k_star
                    order = min(self.ipndm_max_order, j + 1)
                    h_n_1 = float(base_sigmas_np[i] - base_sigmas_np[i - 1]) if j >= 1 else 0.0
                    h_n_2 = float(base_sigmas_np[i - 1] - base_sigmas_np[i - 2]) if j >= 2 else 0.0
                    h_n_3 = float(base_sigmas_np[i - 2] - base_sigmas_np[i - 3]) if j >= 3 else 0.0
                weights = self._compute_ab_weights(h_n, h_n_1, h_n_2, h_n_3, order)
                step_configs.append({
                    "mode": "multistep",
                    "interval_idx": i,
                    "is_early": is_early,
                    "s_curr": s_curr,
                    "s_next": s_next,
                    "dt": h_n,
                    "timestep": t_step,
                    "solver": solver,
                    "order": order,
                    "weights": weights,
                })
        self._step_configs = step_configs
        self.timesteps = torch.tensor(eval_timesteps, dtype=torch.float32, device=target_device)
        self.sigmas = self.base_sigmas
        self.has_corrector = any(cfg["mode"] == "corrector" for cfg in step_configs)
        self._step_index = 0
        self._sample_anchor = None
        self._v0 = None
        self._h = None
        self._history.clear()
        self._boundary_flushed = False

    def _push_history(self, v: torch.Tensor, buffer: List[torch.Tensor]) -> None:
        if len(buffer) >= self.ipndm_max_order - 1:
            buffer.pop(0)
        buffer.append(v.detach())

    def _calc_sde_dispersion(
        self,
        ref_tensor: torch.Tensor,
        v_tot: torch.Tensor,
        s_curr: float,
        s_next: float,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.eta <= 0.0 or self.s_noise <= 0.0 or self.handoff_threshold <= 0.0:
            return torch.zeros_like(ref_tensor)
        if s_curr >= self.handoff_threshold:
            return torch.zeros_like(ref_tensor)
        envelope = max(0.0, min(1.0, (self.handoff_threshold - s_curr) / self.handoff_threshold))
        dt_val = abs(s_next - s_curr)
        renoise = self.eta * math.sqrt(max(dt_val, 1e-8)) * envelope * self.s_noise
        if renoise <= 0.0:
            return torch.zeros_like(ref_tensor)
        gen_dev = generator.device.type if generator is not None else None
        if generator is not None and gen_dev == ref_tensor.device.type:
            noise = torch.randn(ref_tensor.shape, generator=generator, device=ref_tensor.device, dtype=ref_tensor.dtype)
        else:
            noise = torch.randn(ref_tensor.shape, device=ref_tensor.device, dtype=ref_tensor.dtype)
        denom = max(1.0 - s_curr, 1e-3)
        score = -(ref_tensor - s_curr * v_tot) / denom
        drift_comp = 0.5 * (renoise**2) * score
        return drift_comp + (noise * renoise)

    def step(
        self,
        model_output: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        num_passes = len(self._step_configs)
        if self._step_index is None or self._step_index >= num_passes:
            self._step_index = 0
            self._sample_anchor = None
            self._v0 = None
            self._h = None
            self._history.clear()
            self._boundary_flushed = False

        if isinstance(model_output, (tuple, list)):
            v = model_output[0]
        else:
            v = model_output

        cfg = self._step_configs[self._step_index]
        mode = cfg["mode"]
        s_curr = cfg["s_curr"]
        s_next = cfg["s_next"]
        dt = cfg["dt"]
        is_early = cfg["is_early"]
        solver = cfg["solver"]

        if not is_early and not self._boundary_flushed:
            self._history.clear()
            self._boundary_flushed = True

        if mode == "predictor":
            self._sample_anchor = sample.clone()
            self._v0 = v.clone()
            self._h = dt
            dx = dt * v
            prev_sample = sample + dx
        elif mode == "corrector":
            sample_0 = self._sample_anchor if self._sample_anchor is not None else sample
            v0 = self._v0 if self._v0 is not None else v
            dt = self._h if self._h is not None else dt
            if solver in ("heun", "sde_gpu_pp", "multitree", "res_multistep", "multires"):
                dx = (dt / 2.0) * (v0 + v)
            else:
                dx = dt * v0
            if is_early and self.eta > 0.0 and s_next < 1.0:
                v_tot = 0.5 * (v0 + v)
                dispersion = self._calc_sde_dispersion(sample_0, v_tot, s_curr, s_next, generator=generator)
                prev_sample = sample_0 + dx + dispersion
            else:
                prev_sample = sample_0 + dx
            self._sample_anchor = None
            self._v0 = None
            self._h = None
        else:
            c0, c1, c2, c3 = cfg["weights"]
            if solver in ("ipndm", "multitree", "res_multistep", "multires"):
                dx = c0 * v
                if len(self._history) >= 1 and c1 != 0.0:
                    dx = dx + c1 * self._history[-1]
                if len(self._history) >= 2 and c2 != 0.0:
                    dx = dx + c2 * self._history[-2]
                if len(self._history) >= 3 and c3 != 0.0:
                    dx = dx + c3 * self._history[-3]
            else:
                dx = dt * v
            if is_early and self.eta > 0.0 and s_next < 1.0:
                dispersion = self._calc_sde_dispersion(sample, v, s_curr, s_next, generator=generator)
                prev_sample = sample + dx + dispersion
            else:
                prev_sample = sample + dx
            self._push_history(v, self._history)

        self._step_index += 1
        if self._step_index >= num_passes:
            self._step_index = None
            self._sample_anchor = None
            self._v0 = None
            self._h = None
            self._history.clear()
            self._boundary_flushed = False
        return prev_sample

__all__ = [
    "FlowMatchEulerDiscreteScheduler",
    "FlowMatchHeunDiscreteScheduler",
    "FlowMatchIPNDMDiscreteScheduler",
    "BifurcatedFlowMatchScheduler",
]