import math
from typing import Dict, Optional, Tuple, Union
import torch

_SPLITMIX64_C1 = 0x9E3779B97F4A7C15
_SPLITMIX64_C2 = 0xBF58476D1CE4E5B9
_SPLITMIX64_C3 = 0x94D049BB133111EB
_MASK64 = 0xFFFFFFFFFFFFFFFF
_MASK63 = 0x7FFFFFFFFFFFFFFF
_EPS_FP32 = 1.1920929e-7

_GENERATOR_CACHE: Dict[str, torch.Generator] = {}


def splitmix64(seed: int) -> int:
    z = (seed + _SPLITMIX64_C1) & _MASK64
    z = ((z ^ (z >> 30)) * _SPLITMIX64_C2) & _MASK64
    z = ((z ^ (z >> 27)) * _SPLITMIX64_C3) & _MASK64
    return z ^ (z >> 31)


def derive_stage_keys(master_seed: int) -> Tuple[int, int, int]:
    base = master_seed & _MASK64
    k_ar = splitmix64(base ^ 0x5A5A5A5A5A5A5A5A)
    k_rvq = splitmix64(base ^ 0xA5A5A5A5A5A5A5A5)
    k_dit = splitmix64(base ^ 0x3C3C3C3C3C3C3C3C)
    return k_ar, k_rvq, k_dit


def derive_effective_seed(seed: int, stream_id: int = 0, offset: int = 0) -> int:
    h = splitmix64((seed ^ (stream_id * _SPLITMIX64_C1)) & _MASK64)
    h = splitmix64((h + offset + _SPLITMIX64_C2) & _MASK64)
    return int(h & _MASK63)


def get_device_generator(device: torch.device, seed: int) -> torch.Generator:
    dev_key = f"{device.type}:{device.index if device.index is not None else 0}"
    gen = _GENERATOR_CACHE.get(dev_key)
    if gen is None:
        gen = torch.Generator(device=device)
        _GENERATOR_CACHE[dev_key] = gen
    gen.manual_seed(seed)
    return gen


def philox_uniform(
    shape: Union[Tuple[int, ...], torch.Size],
    seed: int,
    stream_id: int = 0,
    offset: int = 0,
    device: Union[str, torch.device] = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    target_device = torch.device(device) if isinstance(device, str) else device
    eff_seed = derive_effective_seed(seed, stream_id, offset)
    gen = get_device_generator(target_device, eff_seed)
    return torch.rand(shape, generator=gen, device=target_device, dtype=dtype)


def philox_randn(
    shape: Union[Tuple[int, ...], torch.Size],
    seed: int,
    stream_id: int = 0,
    offset: int = 0,
    device: Union[str, torch.device] = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    target_device = torch.device(device) if isinstance(device, str) else device
    eff_seed = derive_effective_seed(seed, stream_id, offset)
    gen = get_device_generator(target_device, eff_seed)
    return torch.randn(shape, generator=gen, device=target_device, dtype=dtype)


def philox_gumbel(
    shape: Union[Tuple[int, ...], torch.Size],
    seed: int,
    stream_id: int = 0,
    offset: int = 0,
    device: Union[str, torch.device] = torch.device("cpu"),
) -> torch.Tensor:
    target_device = torch.device(device) if isinstance(device, str) else device
    eff_seed = derive_effective_seed(seed, stream_id, offset)
    gen = get_device_generator(target_device, eff_seed)
    u = torch.rand(shape, generator=gen, device=target_device, dtype=torch.float32).clamp(1e-12, 1.0 - _EPS_FP32)
    return -torch.log(-torch.log(u))


def deterministic_gumbel_sample_vector(
    logits: torch.Tensor,
    temperature: float,
    seed: int,
    stream_id: int,
) -> torch.Tensor:
    values = torch.nan_to_num(logits.to(torch.float32), nan=-1e9, posinf=1e9, neginf=-1e9)
    target_device = logits.device
    eff_seed = derive_effective_seed(seed, stream_id, 0)
    gen = get_device_generator(target_device, eff_seed)
    u = torch.rand(values.shape, generator=gen, device=target_device, dtype=torch.float32).clamp(1e-12, 1.0 - _EPS_FP32)
    gumbel_noise = -torch.log(-torch.log(u))
    temp_eff = max(float(temperature), 1e-4)
    perturbed = (values / temp_eff) + gumbel_noise
    return torch.argmax(perturbed, dim=-1)


class Philox4x32Engine:
    @classmethod
    def generate_uniform(
        cls,
        numel: int,
        seed: int,
        stream_id: int = 0,
        offset: int = 0,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> torch.Tensor:
        if numel <= 0:
            return torch.empty(0, device=device, dtype=torch.float32)
        return philox_uniform((numel,), seed=seed, stream_id=stream_id, offset=offset, device=device)

    @classmethod
    def generate_normal(
        cls,
        numel: int,
        seed: int,
        stream_id: int = 0,
        offset: int = 0,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> torch.Tensor:
        if numel <= 0:
            return torch.empty(0, device=device, dtype=torch.float32)
        return philox_randn((numel,), seed=seed, stream_id=stream_id, offset=offset, device=device)


__all__ = [
    "Philox4x32Engine",
    "philox_uniform",
    "philox_randn",
    "philox_gumbel",
    "splitmix64",
    "derive_stage_keys",
    "derive_effective_seed",
    "deterministic_gumbel_sample_vector",
]