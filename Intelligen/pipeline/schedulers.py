import math
from typing import Optional, Union, List
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

        # Default linear sigmas from 1.0 down to 1.0 / steps
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=np.float32)

        # Dynamic or static shift
        if self.shift != 1.0:
            sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)

        target_device = torch.device(device) if device is not None else torch.device("cpu")
        sigmas_tensor = torch.from_numpy(sigmas).to(dtype=torch.float32, device=target_device)

        # Inverted schedule: 0 -> 1 (0 = pure noise, 1 = data)
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