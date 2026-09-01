import math
from typing import Optional, Tuple, Union
import torch

_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85
_TWO_POW_32_RECIP = 1.0 / 4294967296.0
_SPLITMIX64_C1 = 0x9E3779B97F4A7C15
_SPLITMIX64_C2 = 0xBF58476D1CE4E5B9
_SPLITMIX64_C3 = 0x94D049BB133111EB
_MASK64 = 0xFFFFFFFFFFFFFFFF
_MASK32 = 0xFFFFFFFF
_EPS_FP32 = 1.1920929e-7


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


class Philox4x32Engine:
    @staticmethod
    def _philox4x32_10_step(
        c0: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        k0: int,
        k1: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        curr_k0 = k0 & _MASK32
        curr_k1 = k1 & _MASK32
        curr_c0, curr_c1, curr_c2, curr_c3 = c0, c1, c2, c3

        for _ in range(10):
            prod0 = curr_c0 * _PHILOX_M0
            hi0 = (prod0 >> 32) & _MASK32
            lo0 = prod0 & _MASK32

            prod1 = curr_c2 * _PHILOX_M1
            hi1 = (prod1 >> 32) & _MASK32
            lo1 = prod1 & _MASK32

            next_c0 = (hi1 ^ curr_k0 ^ curr_c1) & _MASK32
            next_c1 = lo1
            next_c2 = (hi0 ^ curr_k1 ^ curr_c3) & _MASK32
            next_c3 = lo0

            curr_c0, curr_c1, curr_c2, curr_c3 = next_c0, next_c1, next_c2, next_c3
            curr_k0 = (curr_k0 + _PHILOX_W0) & _MASK32
            curr_k1 = (curr_k1 + _PHILOX_W1) & _MASK32

        return curr_c0, curr_c1, curr_c2, curr_c3

    @classmethod
    def generate_uniform(
        cls,
        numel: int,
        seed: int,
        stream_id: int = 0,
        offset: int = 0,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        if numel <= 0:
            return torch.empty(0, device=device, dtype=torch.float32)

        num_vectors = (numel + 3) // 4
        indices = torch.arange(
            offset, offset + num_vectors, device=device, dtype=torch.int64
        )

        c0 = (indices * 4) & _MASK32
        c1 = ((indices * 4) >> 32) & _MASK32
        c2 = torch.full_like(indices, stream_id & _MASK32)
        c3 = torch.full_like(indices, (stream_id >> 32) & _MASK32)

        k0 = seed & _MASK32
        k1 = (seed >> 32) & _MASK32

        o0, o1, o2, o3 = cls._philox4x32_10_step(c0, c1, c2, c3, k0, k1)

        u0 = (o0.to(torch.float32) + 0.5) * _TWO_POW_32_RECIP
        u1 = (o1.to(torch.float32) + 0.5) * _TWO_POW_32_RECIP
        u2 = (o2.to(torch.float32) + 0.5) * _TWO_POW_32_RECIP
        u3 = (o3.to(torch.float32) + 0.5) * _TWO_POW_32_RECIP

        packed = torch.stack((u0, u1, u2, u3), dim=-1).view(-1)
        return packed[:numel]

    @classmethod
    def generate_normal(
        cls,
        numel: int,
        seed: int,
        stream_id: int = 0,
        offset: int = 0,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        if numel <= 0:
            return torch.empty(0, device=device, dtype=torch.float32)

        num_pairs = (numel + 3) // 4
        indices = torch.arange(
            offset, offset + num_pairs, device=device, dtype=torch.int64
        )

        c0 = (indices * 4) & _MASK32
        c1 = ((indices * 4) >> 32) & _MASK32
        c2 = torch.full_like(indices, stream_id & _MASK32)
        c3 = torch.full_like(indices, (stream_id >> 32) & _MASK32)

        k0 = seed & _MASK32
        k1 = (seed >> 32) & _MASK32

        o0, o1, o2, o3 = cls._philox4x32_10_step(c0, c1, c2, c3, k0, k1)

        u0 = torch.clamp(
            (o0.to(torch.float32) + 0.5) * _TWO_POW_32_RECIP, 1e-12, 1.0 - _EPS_FP32
        )
        u1 = (o1.to(torch.float32) + 0.5) * _TWO_POW_32_RECIP
        u2 = torch.clamp(
            (o2.to(torch.float32) + 0.5) * _TWO_POW_32_RECIP, 1e-12, 1.0 - _EPS_FP32
        )
        u3 = (o3.to(torch.float32) + 0.5) * _TWO_POW_32_RECIP

        radius0 = torch.sqrt(-2.0 * torch.log(u0))
        theta0 = (2.0 * math.pi) * u1
        z0 = radius0 * torch.cos(theta0)
        z1 = radius0 * torch.sin(theta0)

        radius1 = torch.sqrt(-2.0 * torch.log(u2))
        theta1 = (2.0 * math.pi) * u3
        z2 = radius1 * torch.cos(theta1)
        z3 = radius1 * torch.sin(theta1)

        packed = torch.stack((z0, z1, z2, z3), dim=-1).view(-1)
        return packed[:numel]


def philox_uniform(
    shape: Union[Tuple[int, ...], torch.Size],
    seed: int,
    stream_id: int = 0,
    offset: int = 0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    raw = Philox4x32Engine.generate_uniform(
        numel=total_elements,
        seed=seed,
        stream_id=stream_id,
        offset=offset,
        device=device,
    )
    return raw.view(shape).to(dtype=dtype)


def philox_randn(
    shape: Union[Tuple[int, ...], torch.Size],
    seed: int,
    stream_id: int = 0,
    offset: int = 0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    raw = Philox4x32Engine.generate_normal(
        numel=total_elements,
        seed=seed,
        stream_id=stream_id,
        offset=offset,
        device=device,
    )
    return raw.view(shape).to(dtype=dtype)


def deterministic_gumbel_top_k_sample(
    logits: torch.Tensor,
    top_k: int,
    seed: int,
    stream_id: int,
    step_offset: int = 0,
) -> torch.Tensor:
    values = torch.nan_to_num(logits.to(torch.float32), nan=-1e9, posinf=1e9, neginf=-1e9)
    k = min(top_k, values.shape[-1])
    topk_values, topk_indices = torch.topk(values, k, dim=-1)

    uniform = philox_uniform(
        shape=topk_values.shape,
        seed=seed,
        stream_id=stream_id,
        offset=step_offset,
        device=logits.device,
        dtype=torch.float32,
    )

    uniform_clamped = torch.clamp(uniform, 1e-12, 1.0 - _EPS_FP32)
    gumbel_noise = -torch.log(-torch.log(uniform_clamped))
    perturbed = topk_values + gumbel_noise
    best_idx = torch.argmax(perturbed, dim=-1, keepdim=True)
    return torch.gather(topk_indices, -1, best_idx).squeeze(-1)