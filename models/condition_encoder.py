from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniMaxMusic3ConditionEncoder(nn.Module):
    def __init__(
        self,
        condition_hidden_dim: int = 4096,
        num_condition_layers: int = 8,
        out_dim: int = 2048,
        input_sampling_rate: int = 24000,
        input_hop_length: int = 960,
        output_sampling_rate: int = 44100,
        output_hop_length: int = 512,
    ):
        super().__init__()
        self.condition_hidden_dim = condition_hidden_dim
        self.num_condition_layers = num_condition_layers
        self.out_dim = out_dim
        self.input_sampling_rate = input_sampling_rate
        self.input_hop_length = input_hop_length
        self.output_sampling_rate = output_sampling_rate
        self.output_hop_length = output_hop_length
        self.layer_weight_logits = nn.Parameter(torch.zeros(num_condition_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))
        self.proj = nn.Conv1d(condition_hidden_dim, out_dim, kernel_size=3, padding=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask_layer_0: bool = False,
        return_bifurcated: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, num_frames, _ = hidden_states.shape
        num_layers = self.num_condition_layers
        h = hidden_states.transpose(1, 2).reshape(
            batch_size, num_layers, self.condition_hidden_dim, num_frames
        )
        layer_weights = torch.softmax(self.layer_weight_logits, dim=0).to(h.dtype)

        latent_length = max(
            1,
            int(
                num_frames
                * self.output_sampling_rate
                / self.input_sampling_rate
                * self.input_hop_length
                / self.output_hop_length
            ),
        )

        if return_bifurcated:
            h_full = torch.einsum("blht,l->bht", h, layer_weights)
            h_inst = torch.einsum("blht,l->bht", h[:, 1:], layer_weights[1:])
            h_batched = torch.cat([h_full, h_inst], dim=0)
            h_batched = self.layer_scale.to(h_batched.dtype) * h_batched
            h_batched = self.proj(h_batched)
            h_batched = F.interpolate(h_batched, size=latent_length, mode="nearest")
            out = h_batched.transpose(1, 2)
            return out[:batch_size], out[batch_size:]

        if mask_layer_0:
            h_proj = torch.einsum("blht,l->bht", h[:, 1:], layer_weights[1:])
        else:
            h_proj = torch.einsum("blht,l->bht", h, layer_weights)

        h_proj = self.layer_scale.to(h_proj.dtype) * h_proj
        h_proj = self.proj(h_proj)
        h_proj = F.interpolate(h_proj, size=latent_length, mode="nearest")
        return h_proj.transpose(1, 2)