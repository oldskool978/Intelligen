import math
from typing import Optional, Union, List, Tuple
import numpy as np
import torch


class FlowMatchEulerDiscreteScheduler:
    def __init__(self, shift: float = 1.0, num_train_timesteps: int = 1):
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps
        self.timesteps: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self._step_index: Optional[int] = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[str, torch.device]] = None,
        shift: Optional[float] = None,
    ) -> None:
        if shift is not None:
            self.shift = shift
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=np.float32)
        if self.shift != 1.0:
            sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        sigmas_tensor = torch.from_numpy(sigmas).to(dtype=torch.float32, device=target_device)
        sigmas_tensor = 1.0 - sigmas_tensor
        self.timesteps = sigmas_tensor * self.num_train_timesteps
        self.sigmas = torch.cat([sigmas_tensor, torch.ones(1, device=target_device)])
        self._step_index = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
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
        instrumental_solver: str = "heun",
        vocal_solver: str = "heun",
        shift: float = 1.0,
        num_train_timesteps: int = 1,
        ipndm_max_order: int = 4,
        eta: float = 0.0,
        s_noise: float = 1.0,
    ):
        self.instrumental_solver = instrumental_solver.lower()
        self.vocal_solver = vocal_solver.lower()
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps
        self.ipndm_max_order = max(1, min(4, int(ipndm_max_order)))
        self.eta = max(0.0, min(1.0, float(eta)))
        self.s_noise = max(0.0, float(s_noise))

        self.has_corrector = (
            self.instrumental_solver in ("heun", "sde_gpu_pp")
            or self.vocal_solver in ("heun", "sde_gpu_pp")
        )
        self.base_sigmas: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self._sigmas_cpu: Optional[np.ndarray] = None
        self._step_index: Optional[int] = None

        self._sample_anchor: Optional[torch.Tensor] = None
        self._v0_inst: Optional[torch.Tensor] = None
        self._v0_vocal: Optional[torch.Tensor] = None
        self._v0_uncond: Optional[torch.Tensor] = None
        self._h: Optional[torch.Tensor] = None

        self._ipndm_buffer_inst: List[torch.Tensor] = []
        self._ipndm_buffer_vocal: List[torch.Tensor] = []

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[str, torch.device]] = None,
        shift: Optional[float] = None,
    ) -> None:
        if shift is not None:
            self.shift = shift
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=np.float32)
        if self.shift != 1.0:
            sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        sigmas_tensor = torch.from_numpy(sigmas).to(dtype=torch.float32, device=target_device)
        sigmas_tensor = 1.0 - sigmas_tensor
        self.base_sigmas = torch.cat([sigmas_tensor, torch.ones(1, device=target_device)])
        self._sigmas_cpu = self.base_sigmas.detach().cpu().numpy()

        self.has_corrector = (
            self.instrumental_solver in ("heun", "sde_gpu_pp")
            or self.vocal_solver in ("heun", "sde_gpu_pp")
        )
        if self.has_corrector:
            num_intervals = len(self.base_sigmas) - 1
            heun_sigmas = []
            for i in range(num_intervals):
                s_curr = self.base_sigmas[i]
                s_next = self.base_sigmas[i + 1]
                heun_sigmas.extend([s_curr, s_next])
            heun_sigmas.append(self.base_sigmas[-1])
            self.sigmas = torch.stack(heun_sigmas).to(device=target_device)
            self.timesteps = (self.sigmas[:-1] * self.num_train_timesteps).clone()
        else:
            self.sigmas = self.base_sigmas
            self.timesteps = (sigmas_tensor * self.num_train_timesteps).clone()

        self._step_index = None
        self._sample_anchor = None
        self._v0_inst = None
        self._v0_vocal = None
        self._v0_uncond = None
        self._h = None
        self._ipndm_buffer_inst.clear()
        self._ipndm_buffer_vocal.clear()

    def _calc_ab_dx(
        self,
        v_cur: torch.Tensor,
        buffer: List[torch.Tensor],
        interval_idx: int,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        order = min(self.ipndm_max_order, interval_idx + 1)
        if order == 1 or len(buffer) == 0:
            return dt * v_cur
        elif order == 2 or len(buffer) == 1:
            h_n = float(self._sigmas_cpu[interval_idx + 1] - self._sigmas_cpu[interval_idx])
            h_n_1 = float(self._sigmas_cpu[interval_idx] - self._sigmas_cpu[interval_idx - 1])
            r = h_n / max(h_n_1, 1e-12)
            c1 = (2.0 + r) / 2.0
            c2 = -r / 2.0
            return dt * (c1 * v_cur + c2 * buffer[-1])
        elif order == 3 or len(buffer) == 2:
            h_n = float(self._sigmas_cpu[interval_idx + 1] - self._sigmas_cpu[interval_idx])
            h_n_1 = float(self._sigmas_cpu[interval_idx] - self._sigmas_cpu[interval_idx - 1])
            h_n_2 = float(self._sigmas_cpu[interval_idx - 1] - self._sigmas_cpu[interval_idx - 2])
            temp = (
                1.0
                - (h_n / (3.0 * max(h_n + h_n_1, 1e-12)))
                * ((h_n * (h_n + h_n_1)) / max(h_n_1 * (h_n_1 + h_n_2), 1e-12))
            ) / 2.0
            c1 = (2.0 + (h_n / max(h_n_1, 1e-12))) / 2.0 + temp
            c2 = -(h_n / max(h_n_1, 1e-12)) / 2.0 - (1.0 + h_n_1 / max(h_n_2, 1e-12)) * temp
            c3 = temp * (h_n_1 / max(h_n_2, 1e-12))
            return dt * (c1 * v_cur + c2 * buffer[-1] + c3 * buffer[-2])
        else:
            h_n = float(self._sigmas_cpu[interval_idx + 1] - self._sigmas_cpu[interval_idx])
            h_n_1 = float(self._sigmas_cpu[interval_idx] - self._sigmas_cpu[interval_idx - 1])
            h_n_2 = float(self._sigmas_cpu[interval_idx - 1] - self._sigmas_cpu[interval_idx - 2])
            h_n_3 = float(self._sigmas_cpu[interval_idx - 2] - self._sigmas_cpu[interval_idx - 3])
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
            return dt * (
                c1 * v_cur
                + c2 * buffer[-1]
                + c3 * buffer[-2]
                + c4 * buffer[-3]
            )

    def _calc_am_dx(
        self,
        v_next: torch.Tensor,
        v_curr: torch.Tensor,
        buffer: List[torch.Tensor],
        interval_idx: int,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        h_n = float(self._sigmas_cpu[interval_idx + 1] - self._sigmas_cpu[interval_idx])
        trapezoid = (dt / 2.0) * (v_curr + v_next)
        if len(buffer) == 0 or interval_idx == 0:
            return trapezoid
        h_n_1 = float(self._sigmas_cpu[interval_idx] - self._sigmas_cpu[interval_idx - 1])
        v_prev = buffer[-1]
        scale = (h_n**2) / (6.0 * max(h_n + h_n_1, 1e-12))
        curvature = (v_next - v_curr) - (h_n / max(h_n_1, 1e-12)) * (v_curr - v_prev)
        return trapezoid - scale * curvature

    def _calc_sde_pp_dx(
        self,
        v_guided: torch.Tensor,
        v_uncond: Optional[torch.Tensor],
        t: float,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        decay = max(0.0, 1.0 - t)
        if v_uncond is None:
            return dt * (decay * v_guided)
        v_eff = torch.lerp(v_uncond, v_guided, decay)
        return dt * v_eff

    def _calc_sde_dispersion(
        self,
        ref_tensor: torch.Tensor,
        t_curr: float,
        t_next: float,
    ) -> torch.Tensor:
        sigma_t = max(0.0, 1.0 - t_next)
        dt_val = abs(t_next - t_curr)
        renoise = self.eta * math.sqrt(max(dt_val, 1e-8)) * sigma_t * self.s_noise
        if renoise > 0.0:
            return torch.randn_like(ref_tensor) * renoise
        return torch.zeros_like(ref_tensor)

    def _update_buffer(self, v: torch.Tensor, buffer: List[torch.Tensor]) -> None:
        if len(buffer) == self.ipndm_max_order - 1:
            buffer.pop(0)
        buffer.append(v.detach())

    def step(
        self,
        model_output: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
    ) -> torch.Tensor:
        num_steps = len(self.timesteps)
        if self._step_index is None or self._step_index >= num_steps:
            self._step_index = 0
            self._sample_anchor = None
            self._v0_inst = None
            self._v0_vocal = None
            self._v0_uncond = None
            self._h = None
            self._ipndm_buffer_inst.clear()
            self._ipndm_buffer_vocal.clear()
            if self._sigmas_cpu is None:
                self._sigmas_cpu = self.base_sigmas.detach().cpu().numpy()

        if isinstance(model_output, (tuple, list)):
            v_inst = model_output[0]
            v_vocal = model_output[1]
            v_uncond = model_output[2] if len(model_output) > 2 else model_output[0]
        else:
            v_inst = model_output
            v_vocal = torch.zeros_like(model_output)
            v_uncond = model_output

        idx = self._step_index
        has_sde = (self.instrumental_solver == "sde_gpu_pp" or self.vocal_solver == "sde_gpu_pp")

        if self.has_corrector:
            is_predictor = (idx % 2 == 0)
            interval_idx = idx // 2
            t_curr = float(self._sigmas_cpu[interval_idx])
            t_next = float(self._sigmas_cpu[interval_idx + 1])
            s_curr = self.base_sigmas[interval_idx]
            s_next = self.base_sigmas[interval_idx + 1]
            dt = s_next - s_curr

            if is_predictor:
                self._sample_anchor = sample.clone()
                self._v0_inst = v_inst.clone()
                self._v0_vocal = v_vocal.clone()
                self._v0_uncond = v_uncond.clone()
                self._h = dt

                if self.instrumental_solver == "ipndm":
                    dx_inst = self._calc_ab_dx(
                        v_inst, self._ipndm_buffer_inst, interval_idx, dt
                    )
                elif self.instrumental_solver == "sde_gpu_pp":
                    dx_inst = self._calc_sde_pp_dx(v_inst, v_uncond, t_curr, dt)
                else:
                    dx_inst = dt * v_inst

                if self.vocal_solver == "ipndm":
                    dx_vocal = self._calc_ab_dx(
                        v_vocal, self._ipndm_buffer_vocal, interval_idx, dt
                    )
                elif self.vocal_solver == "sde_gpu_pp":
                    dx_vocal = self._calc_sde_pp_dx(v_vocal, None, t_curr, dt)
                else:
                    dx_vocal = dt * v_vocal

                prev_sample = sample + dx_inst + dx_vocal
            else:
                v0_inst = self._v0_inst if self._v0_inst is not None else v_inst
                v0_vocal = self._v0_vocal if self._v0_vocal is not None else v_vocal
                v0_uncond = self._v0_uncond if self._v0_uncond is not None else v_uncond
                sample_0 = self._sample_anchor if self._sample_anchor is not None else sample
                dt = self._h if self._h is not None else dt
                t_mid = 0.5 * (t_curr + t_next)

                if self.instrumental_solver == "heun":
                    dx_inst = (dt / 2.0) * (v0_inst + v_inst)
                elif self.instrumental_solver == "ipndm":
                    dx_inst = self._calc_am_dx(
                        v_inst, v0_inst, self._ipndm_buffer_inst, interval_idx, dt
                    )
                elif self.instrumental_solver == "sde_gpu_pp":
                    v_inst_avg = 0.5 * (v0_inst + v_inst)
                    v_uncond_avg = 0.5 * (v0_uncond + v_uncond)
                    dx_inst = self._calc_sde_pp_dx(v_inst_avg, v_uncond_avg, t_mid, dt)
                else:
                    dx_inst = dt * v0_inst

                if self.vocal_solver == "heun":
                    dx_vocal = (dt / 2.0) * (v0_vocal + v_vocal)
                elif self.vocal_solver == "ipndm":
                    dx_vocal = self._calc_am_dx(
                        v_vocal, v0_vocal, self._ipndm_buffer_vocal, interval_idx, dt
                    )
                elif self.vocal_solver == "sde_gpu_pp":
                    v_vocal_avg = 0.5 * (v0_vocal + v_vocal)
                    dx_vocal = self._calc_sde_pp_dx(v_vocal_avg, None, t_mid, dt)
                else:
                    dx_vocal = dt * v0_vocal

                if self.instrumental_solver == "ipndm":
                    self._update_buffer(v0_inst, self._ipndm_buffer_inst)
                if self.vocal_solver == "ipndm":
                    self._update_buffer(v0_vocal, self._ipndm_buffer_vocal)

                if has_sde and self.eta > 0.0 and t_next < 1.0:
                    dispersion = self._calc_sde_dispersion(sample_0, t_curr, t_next)
                    prev_sample = sample_0 + dx_inst + dx_vocal + dispersion
                else:
                    prev_sample = sample_0 + dx_inst + dx_vocal

                self._sample_anchor = None
                self._v0_inst = None
                self._v0_vocal = None
                self._v0_uncond = None
                self._h = None
        else:
            interval_idx = idx
            t_curr = float(self._sigmas_cpu[interval_idx])
            t_next = float(self._sigmas_cpu[interval_idx + 1])
            s_curr = self.sigmas[idx]
            s_next = self.sigmas[idx + 1]
            dt = s_next - s_curr

            if self.instrumental_solver == "ipndm":
                dx_inst = self._calc_ab_dx(
                    v_inst, self._ipndm_buffer_inst, interval_idx, dt
                )
                self._update_buffer(v_inst, self._ipndm_buffer_inst)
            elif self.instrumental_solver == "sde_gpu_pp":
                dx_inst = self._calc_sde_pp_dx(v_inst, v_uncond, t_curr, dt)
            else:
                dx_inst = dt * v_inst

            if self.vocal_solver == "ipndm":
                dx_vocal = self._calc_ab_dx(
                    v_vocal, self._ipndm_buffer_vocal, interval_idx, dt
                )
                self._update_buffer(v_vocal, self._ipndm_buffer_vocal)
            elif self.vocal_solver == "sde_gpu_pp":
                dx_vocal = self._calc_sde_pp_dx(v_vocal, None, t_curr, dt)
            else:
                dx_vocal = dt * v_vocal

            if has_sde and self.eta > 0.0 and t_next < 1.0:
                dispersion = self._calc_sde_dispersion(sample, t_curr, t_next)
                prev_sample = sample + dx_inst + dx_vocal + dispersion
            else:
                prev_sample = sample + dx_inst + dx_vocal

        self._step_index += 1
        if self._step_index >= num_steps:
            self._step_index = None
            self._sample_anchor = None
            self._v0_inst = None
            self._v0_vocal = None
            self._v0_uncond = None
            self._h = None
            self._ipndm_buffer_inst.clear()
            self._ipndm_buffer_vocal.clear()

        return prev_sample


__all__ = [
    "FlowMatchEulerDiscreteScheduler",
    "FlowMatchHeunDiscreteScheduler",
    "FlowMatchIPNDMDiscreteScheduler",
    "BifurcatedFlowMatchScheduler",
]