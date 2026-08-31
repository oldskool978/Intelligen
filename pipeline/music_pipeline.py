import math
from typing import Optional, Tuple, List, Callable, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import Qwen3ForCausalLM
except ImportError:
    Qwen3ForCausalLM = AutoModelForCausalLM

from models.depth_decoder import MiniMaxMusic3RVQDepthDecoder
from models.condition_encoder import MiniMaxMusic3ConditionEncoder
from models.transformer import MiniMaxMusic3Transformer1DModel
from models.vocoder import MiniMaxMusic3Vocoder
from pipeline.prompt_compiler import (
    build_text_ids,
    _AUDIO_CODE_OFFSET,
    _AUDIO_END_TOKEN_ID,
    _SEMANTIC_VOCAB_SIZE,
)
from pipeline.schedulers import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
)

_CHUNK_FRAMES = 200
_CHUNK_HOP = 100
_OVERLAP_LATENT_LENGTH = 172
_CROP_LEFT_LATENT = 86
_CROP_RIGHT_LATENT = 344 - 86
_MAX_AUDIO_FRAMES = 9_000


def sample_top_k(
    logits: torch.Tensor,
    top_k: int = 50,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    k = min(top_k, values.shape[-1])
    threshold = torch.topk(values, k, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    probs = torch.nan_to_num(F.softmax(values, dim=-1), nan=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sample_device = generator.device if generator is not None else probs.device
    return torch.multinomial(probs.to(sample_device), 1, generator=generator).squeeze(-1).to(probs.device)


def embed_audio_frame(
    language_model: Qwen3ForCausalLM,
    rvq_depth_decoder: MiniMaxMusic3RVQDepthDecoder,
    frame_codes: torch.Tensor,
) -> torch.Tensor:
    num_codebooks = rvq_depth_decoder.num_codebooks
    embeds = language_model.model.embed_tokens(frame_codes[:, :1] + _AUDIO_CODE_OFFSET)
    offsets = (
        torch.arange(num_codebooks - 1, device=frame_codes.device) * rvq_depth_decoder.audio_vocab_size
    ).unsqueeze(0)
    extra = rvq_depth_decoder.audio_embeddings(frame_codes[:, 1:] + offsets).sum(dim=1, keepdim=True)
    embeds = embeds + extra.to(embeds.dtype)
    return embeds * (num_codebooks**-0.5)


def generate_depth_codes(
    language_model: Qwen3ForCausalLM,
    rvq_depth_decoder: MiniMaxMusic3RVQDepthDecoder,
    last_hidden: torch.Tensor,
    semantic_code: torch.Tensor,
    cfg_scale: float,
    generator: Optional[torch.Generator],
    top_k: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_codebooks = rvq_depth_decoder.num_codebooks
    sequence = [rvq_depth_decoder.projection(last_hidden).unsqueeze(1)]
    code_embed = language_model.model.embed_tokens(semantic_code + _AUDIO_CODE_OFFSET)
    sequence.append(rvq_depth_decoder.projection(code_embed).unsqueeze(1))
    codes = [semantic_code]
    hidden_parts = []
    for index in range(1, num_codebooks):
        hidden = rvq_depth_decoder(torch.cat(sequence, dim=1))[:, -1]
        hidden_parts.append(hidden[:1])
        logits = rvq_depth_decoder.audio_heads[index - 1](hidden)
        conditional, unconditional = logits[:1].float(), logits[1:2].float()
        guided = unconditional + (conditional - unconditional) * cfg_scale
        code = sample_top_k(
            guided,
            top_k=top_k,
            generator=generator,
        ).repeat(2)
        codes.append(code)
        if index < num_codebooks - 1:
            embed = rvq_depth_decoder.audio_embeddings(
                code + (index - 1) * rvq_depth_decoder.audio_vocab_size
            )
            sequence.append(rvq_depth_decoder.projection(embed).unsqueeze(1))
    return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)


class MiniMaxMusic3Pipeline:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        language_model: Qwen3ForCausalLM,
        rvq_depth_decoder: MiniMaxMusic3RVQDepthDecoder,
        condition_encoder: MiniMaxMusic3ConditionEncoder,
        transformer: MiniMaxMusic3Transformer1DModel,
        vocoder: MiniMaxMusic3Vocoder,
        sampling_rate: int = 44100,
        frame_rate: float = 25.0,
        latent_hop_length: int = 512,
        num_channels_latents: int = 128,
    ):
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.rvq_depth_decoder = rvq_depth_decoder
        self.condition_encoder = condition_encoder
        self.transformer = transformer
        self.vocoder = vocoder
        self.sampling_rate = sampling_rate
        self.frame_rate = frame_rate
        self.latent_hop_length = latent_hop_length
        self.num_channels_latents = num_channels_latents

    @torch.no_grad()
    def generate_stage1_autoregressive(
        self,
        text_ids: torch.Tensor,
        audio_duration: float,
        generator: Optional[torch.Generator] = None,
        cfg_scale: float = 1.5,
        cfg_top_k: int = 50,
        sampling_top_k: int = 50,
        show_progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        max_frames = min(int(audio_duration * self.frame_rate), _MAX_AUDIO_FRAMES)
        if max_frames <= 0:
            raise ValueError(f"`audio_duration` {audio_duration} is shorter than one frame.")

        text_embeds = self.language_model.model.embed_tokens(text_ids)
        output = self.language_model.model(inputs_embeds=text_embeds, use_cache=True)
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]
        del text_embeds, output

        vocab_mask = torch.ones(
            self.language_model.config.vocab_size, dtype=torch.bool, device=text_ids.device
        )
        vocab_mask[_AUDIO_CODE_OFFSET : _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE] = False
        vocab_mask[_AUDIO_END_TOKEN_ID] = False

        frame_hiddens = []
        pbar = None
        if show_progress and progress_callback is None:
            pbar = tqdm(
                total=max_frames,
                desc="Stage 1 [Autoregressive LM]",
                dynamic_ncols=True,
            )

        try:
            for frame_index in range(max_frames + 1):
                logits = self.language_model.lm_head(last_hidden).float()
                logits = logits.masked_fill(vocab_mask, -float("inf"))
                conditional, unconditional = logits[0:1], logits[1:2]
                guided = unconditional + (conditional - unconditional) * cfg_scale
                threshold = torch.topk(conditional, cfg_top_k, dim=-1).values[..., -1, None]
                guided = guided.masked_fill(conditional < threshold, -float("inf"))
                guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))

                sampled = sample_top_k(
                    guided,
                    top_k=sampling_top_k,
                    generator=generator,
                )

                if int(sampled.item()) == _AUDIO_END_TOKEN_ID:
                    break

                semantic_code = sampled - _AUDIO_CODE_OFFSET
                frame_codes, depth_hidden = generate_depth_codes(
                    self.language_model,
                    self.rvq_depth_decoder,
                    last_hidden,
                    semantic_code.repeat(2),
                    cfg_scale=cfg_scale,
                    generator=generator,
                    top_k=sampling_top_k,
                )

                if frame_index > 0:
                    frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1))
                    if pbar is not None:
                        pbar.update(1)
                    if progress_callback is not None:
                        progress_callback(len(frame_hiddens), max_frames)
                    if len(frame_hiddens) >= max_frames:
                        break

                feedback = embed_audio_frame(
                    self.language_model, self.rvq_depth_decoder, frame_codes
                )
                output = self.language_model.model(
                    inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True
                )
                past_key_values = output.past_key_values
                last_hidden = output.last_hidden_state[:, -1]
                del feedback, output
        finally:
            if pbar is not None:
                pbar.close()
            del past_key_values, last_hidden, vocab_mask

        if not frame_hiddens:
            raise ValueError("Zero audio frames produced. Prompt triggered termination immediately.")

        return torch.stack(frame_hiddens, dim=1)

    @torch.no_grad()
    def generate_stage2_flow_matching(
        self,
        frame_hiddens: torch.Tensor,
        scheduler: Union[FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler],
        num_inference_steps: int = 42,
        guidance_scale: float = 1.78,
        generator: Optional[torch.Generator] = None,
        latent_shaping_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        show_progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[torch.Tensor]:
        exec_device = device if device is not None else frame_hiddens.device
        num_frames = frame_hiddens.shape[1]
        chunk_starts = (
            [0] if num_frames <= _CHUNK_FRAMES else list(range(0, num_frames - _CHUNK_HOP, _CHUNK_HOP))
        )
        latent_chunks = []
        previous_latent = None
        previous_condition = None

        total_steps = len(chunk_starts) * num_inference_steps
        if isinstance(scheduler, FlowMatchHeunDiscreteScheduler):
            total_steps *= 2

        pbar = None
        if show_progress and progress_callback is None:
            pbar = tqdm(
                total=total_steps,
                desc="Stage 2 [Flow-Matching DiT]",
                dynamic_ncols=True,
            )

        step_counter = 0
        try:
            for chunk_start in chunk_starts:
                chunk_end = min(chunk_start + _CHUNK_FRAMES, num_frames)
                condition = self.condition_encoder(
                    frame_hiddens[:, chunk_start:chunk_end].to(exec_device)
                )
                condition = condition.to(dtype=self.transformer.proj_in.weight.dtype)

                overlap = 0
                if previous_latent is not None:
                    overlap = min(previous_latent.shape[-1], condition.shape[1])
                    condition[:, :overlap] = previous_condition[:, :overlap]

                rand_device = "cpu" if (generator is not None and generator.device.type == "cpu") else exec_device
                latents = torch.randn(
                    (1, self.num_channels_latents, condition.shape[1]),
                    generator=generator,
                    device=rand_device,
                    dtype=condition.dtype,
                ).to(exec_device)

                if latent_shaping_fn is not None:
                    latents = latent_shaping_fn(latents)

                noise_prompt = latents[..., :overlap].clone() if overlap > 0 else None
                scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=exec_device)
                timesteps = scheduler.timesteps
                uncond_condition = torch.zeros_like(condition)

                for t in timesteps:
                    if overlap > 0:
                        time_value = t.to(latents.dtype)
                        latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * time_value) * noise_prompt + (
                            time_value * previous_latent[..., :overlap]
                        )

                    t_expanded = t.expand(latents.shape[0]).to(latents.dtype)
                    batch_latents = torch.cat([latents, latents], dim=0)
                    batch_timesteps = torch.cat([t_expanded, t_expanded], dim=0)
                    batch_cond = torch.cat([condition, uncond_condition], dim=0)

                    batch_pred = self.transformer(
                        hidden_states=batch_latents,
                        timestep=batch_timesteps,
                        encoder_hidden_states=batch_cond,
                    )

                    noise_pred_cond, noise_pred_uncond = batch_pred.chunk(2, dim=0)
                    velocity = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents = scheduler.step(velocity, t, latents)

                    step_counter += 1
                    if pbar is not None:
                        pbar.update(1)
                    if progress_callback is not None:
                        progress_callback(step_counter, total_steps)

                if overlap > 0:
                    latents[..., :overlap] = previous_latent[..., :overlap]

                overlap_start = max(0, latents.shape[-1] - 2 * _OVERLAP_LATENT_LENGTH)
                overlap_end = max(overlap_start, latents.shape[-1] - _OVERLAP_LATENT_LENGTH)
                previous_latent = latents[..., overlap_start:overlap_end]
                previous_condition = condition[:, overlap_start:overlap_end]
                latent_chunks.append(latents)
        finally:
            if pbar is not None:
                pbar.close()

        return latent_chunks

    @torch.no_grad()
    def decode_latents(self, latent_chunks: List[torch.Tensor], batch_size: int = 2) -> torch.Tensor:
        num_chunks = len(latent_chunks)
        if num_chunks == 0:
            return torch.empty((1, 1, 0))

        vocoder_dtype = self.vocoder.dec_in_proj.weight.dtype
        waveform_chunks: List[torch.Tensor] = []
        chunk_idx = 0

        while chunk_idx < num_chunks:
            target_shape = latent_chunks[chunk_idx].shape
            batch_end = chunk_idx + 1
            while (
                batch_end < num_chunks
                and (batch_end - chunk_idx) < batch_size
                and latent_chunks[batch_end].shape == target_shape
            ):
                batch_end += 1

            batch_latents = torch.cat(
                [latent_chunks[k] for k in range(chunk_idx, batch_end)], dim=0
            ).to(vocoder_dtype)
            batch_waveforms = self.vocoder(batch_latents)

            for local_idx, global_idx in enumerate(range(chunk_idx, batch_end)):
                wv = batch_waveforms[local_idx : local_idx + 1]
                left = 0 if global_idx == 0 else _CROP_LEFT_LATENT * self.latent_hop_length
                right = 0 if global_idx == num_chunks - 1 else _CROP_RIGHT_LATENT * self.latent_hop_length
                end_idx = wv.shape[-1] if right == 0 else (wv.shape[-1] - right)
                waveform_chunks.append(wv[..., left:end_idx])

            chunk_idx = batch_end

        return torch.cat(waveform_chunks, dim=-1).float()