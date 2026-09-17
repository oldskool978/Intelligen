import gc
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
    FlowMatchIPNDMDiscreteScheduler,
    BifurcatedFlowMatchScheduler,
)
from pipeline.prng import (
    philox_randn,
    deterministic_gumbel_sample_vector,
    derive_stage_keys,
    derive_effective_seed,
    get_device_generator,
)

_CHUNK_FRAMES = 200
_CHUNK_HOP = 100
_OVERLAP_LATENT_LENGTH = 172
_CROP_LEFT_LATENT = 86
_CROP_RIGHT_LATENT = 344 - 86
_MAX_AUDIO_FRAMES = 9_000
_SEAM_OVERLAP_SAMPLES = 512


def sample_categorical_manifold(
    logits: torch.Tensor,
    top_k: int = 50,
    top_p: float = 1.0,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = None,
    stream_id: int = 0,
) -> torch.Tensor:
    temp_eff = max(float(temperature), 1e-4)
    scaled = logits.to(torch.float32) / temp_eff
    scaled = torch.nan_to_num(scaled, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf"))
    if 0 < top_k < scaled.shape[-1]:
        k_threshold = torch.topk(scaled, top_k, dim=-1).values[..., -1, None]
        scaled = scaled.masked_fill(scaled < k_threshold, -float("inf"))
    if 0.0 < top_p < 1.0:
        max_val = torch.max(scaled, dim=-1, keepdim=True).values
        max_val = torch.where(torch.isneginf(max_val), torch.zeros_like(max_val), max_val)
        exp_logits = torch.exp(scaled - max_val)
        exp_logits = torch.nan_to_num(exp_logits, nan=0.0, posinf=0.0, neginf=0.0)
        probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        scaled = scaled.masked_fill(indices_to_remove, -float("inf"))
    if seed is not None:
        return deterministic_gumbel_sample_vector(
            logits=scaled,
            temperature=1.0,
            seed=seed,
            stream_id=stream_id,
        )
    max_val = torch.max(scaled, dim=-1, keepdim=True).values
    max_val = torch.where(torch.isneginf(max_val), torch.zeros_like(max_val), max_val)
    exp_logits = torch.exp(scaled - max_val)
    exp_logits = torch.nan_to_num(exp_logits, nan=0.0, posinf=0.0, neginf=0.0)
    probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sample_device = generator.device if generator is not None else probs.device
    return torch.multinomial(probs.to(sample_device), 1, generator=generator).to(probs.device).squeeze(-1)


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
    temperature: float,
    generator: Optional[torch.Generator],
    top_k_layers: List[int],
    rvq_seed: Optional[int] = None,
    frame_index: int = 0,
    top_p: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_codebooks = rvq_depth_decoder.num_codebooks
    hidden_size = rvq_depth_decoder.hidden_size
    device = last_hidden.device
    dtype = last_hidden.dtype
    sequence_arena = torch.empty((2, num_codebooks, hidden_size), dtype=dtype, device=device)
    sequence_arena[:, 0] = rvq_depth_decoder.projection(last_hidden)
    code_embed = language_model.model.embed_tokens(semantic_code + _AUDIO_CODE_OFFSET)
    sequence_arena[:, 1] = rvq_depth_decoder.projection(code_embed)
    codes_arena = torch.empty((2, num_codebooks), dtype=semantic_code.dtype, device=device)
    codes_arena[:, 0] = semantic_code
    hidden_parts_arena = torch.empty((1, (num_codebooks - 1) * hidden_size), dtype=dtype, device=device)
    for index in range(1, num_codebooks):
        curr_len = index + 1
        hidden = rvq_depth_decoder(sequence_arena[:, :curr_len])[:, -1]
        hidden_parts_arena[:, (index - 1) * hidden_size : index * hidden_size] = hidden[:1]
        logits = rvq_depth_decoder.audio_heads[index - 1](hidden)
        conditional, unconditional = logits[:1].to(torch.float32), logits[1:2].to(torch.float32)
        guided = unconditional + (conditional - unconditional) * cfg_scale
        layer_k = top_k_layers[index] if index < len(top_k_layers) else 47
        code_stream_id = (0x02 << 32) | (frame_index << 8) | index
        code = sample_categorical_manifold(
            guided,
            top_k=layer_k,
            top_p=top_p,
            temperature=temperature,
            generator=generator,
            seed=rvq_seed,
            stream_id=code_stream_id,
        ).repeat(2)
        codes_arena[:, index] = code
        if index < num_codebooks - 1:
            embed = rvq_depth_decoder.audio_embeddings(
                code + (index - 1) * rvq_depth_decoder.audio_vocab_size
            )
            sequence_arena[:, index + 1] = rvq_depth_decoder.projection(embed)
    return codes_arena, hidden_parts_arena


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
        temperature: float = 0.9192,
        top_p: float = 0.9600,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        cfg_scale: float = 1.5200,
        cfg_top_k: Optional[int] = None,
        top_k_layers: Optional[List[int]] = None,
        show_progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        max_frames = min(int(audio_duration * self.frame_rate), _MAX_AUDIO_FRAMES)
        if max_frames <= 0:
            raise ValueError(f"`audio_duration` {audio_duration} is shorter than one frame.")
        resolved_k_layers = top_k_layers if (top_k_layers and len(top_k_layers) == 8) else [47] * 8
        lm_seed, rvq_seed, _ = derive_stage_keys(seed) if seed is not None else (None, None, None)
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
        frame_hiddens: List[torch.Tensor] = []
        pbar = None
        if show_progress and progress_callback is None:
            total_sec = round(max_frames / self.frame_rate, 1)
            pbar = tqdm(
                total=total_sec,
                desc="Stage 1 [Acoustic LM]",
                unit="s",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n:.1f}/{total_fmt}s [{elapsed}<{remaining}, {rate_fmt}]",
            )
        try:
            for frame_index in range(max_frames + 1):
                logits = self.language_model.lm_head(last_hidden).to(torch.float32)
                logits = logits.masked_fill(vocab_mask, -float("inf"))
                conditional, unconditional = logits[0:1], logits[1:2]
                guided = unconditional + (conditional - unconditional) * cfg_scale
                guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
                lm_stream_id = (0x01 << 32) | frame_index
                sampled = sample_categorical_manifold(
                    guided,
                    top_k=resolved_k_layers[0],
                    top_p=top_p,
                    temperature=temperature,
                    generator=generator,
                    seed=lm_seed,
                    stream_id=lm_stream_id,
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
                    temperature=temperature,
                    generator=generator,
                    top_k_layers=resolved_k_layers,
                    rvq_seed=rvq_seed,
                    frame_index=frame_index,
                    top_p=top_p,
                )
                if frame_index > 0:
                    frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1))
                    if pbar is not None:
                        pbar.update(1.0 / self.frame_rate)
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
        scheduler: Union[
            FlowMatchEulerDiscreteScheduler,
            FlowMatchHeunDiscreteScheduler,
            FlowMatchIPNDMDiscreteScheduler,
            BifurcatedFlowMatchScheduler,
        ],
        num_inference_steps: int = 42,
        guidance_scale: float = 1.7800,
        early_guidance_scale: Optional[float] = None,
        late_guidance_scale: Optional[float] = None,
        handoff_threshold: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        show_progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[torch.Tensor]:
        exec_device = device if device is not None else frame_hiddens.device
        num_frames = frame_hiddens.shape[1]
        chunk_starts = (
            [0] if num_frames <= _CHUNK_FRAMES else list(range(0, num_frames - _CHUNK_HOP, _CHUNK_HOP))
        )
        latent_chunks: List[torch.Tensor] = []
        previous_latent: Optional[torch.Tensor] = None
        previous_condition: Optional[torch.Tensor] = None
        early_cfg = early_guidance_scale if early_guidance_scale is not None else guidance_scale
        late_cfg = late_guidance_scale if late_guidance_scale is not None else early_cfg
        if isinstance(scheduler, BifurcatedFlowMatchScheduler):
            scheduler.set_timesteps(
                num_inference_steps=num_inference_steps,
                device=exec_device,
                handoff_threshold=handoff_threshold,
            )
        else:
            scheduler.set_timesteps(
                num_inference_steps=num_inference_steps,
                device=exec_device,
            )
        nfe_per_chunk = len(scheduler.timesteps)
        total_steps = len(chunk_starts) * nfe_per_chunk
        _, _, dit_seed = derive_stage_keys(seed) if seed is not None else (None, None, None)
        pbar = None
        if show_progress and progress_callback is None:
            pbar = tqdm(
                total=total_steps,
                desc="Stage 2 [Flow-Matching DiT]",
                dynamic_ncols=True,
            )
        step_counter = 0
        try:
            for chunk_idx, chunk_start in enumerate(chunk_starts):
                chunk_end = min(chunk_start + _CHUNK_FRAMES, num_frames)
                raw_chunk = frame_hiddens[:, chunk_start:chunk_end].to(exec_device)
                condition = self.condition_encoder(raw_chunk).contiguous()
                condition = condition.to(dtype=self.transformer.proj_in.weight.dtype)
                overlap = 0
                if previous_latent is not None:
                    overlap = min(previous_latent.shape[-1], condition.shape[1])
                    condition[:, :overlap] = previous_condition[:, :overlap]
                if dit_seed is not None:
                    dit_stream_id = (0x03 << 32) | chunk_idx
                    latents = philox_randn(
                        shape=(1, self.num_channels_latents, condition.shape[1]),
                        seed=dit_seed,
                        stream_id=dit_stream_id,
                        offset=0,
                        device=exec_device,
                        dtype=condition.dtype,
                    )
                    eff_chunk_seed = derive_effective_seed(dit_seed, dit_stream_id, 1)
                    chunk_gen = torch.Generator(device=exec_device).manual_seed(eff_chunk_seed)
                else:
                    rand_device = "cpu" if (generator is not None and generator.device.type == "cpu") else exec_device
                    latents = torch.randn(
                        (1, self.num_channels_latents, condition.shape[1]),
                        generator=generator,
                        device=rand_device,
                        dtype=condition.dtype,
                    ).to(exec_device)
                    chunk_gen = generator if (generator is not None and generator.device.type == exec_device.type) else None
                noise_prompt = latents[..., :overlap].clone() if overlap > 0 else None
                scheduler.reset()
                timesteps = scheduler.timesteps
                uncond_condition = torch.zeros_like(condition)
                batch_cond = torch.cat([condition, uncond_condition], dim=0)
                batch_latents_2 = torch.empty(
                    (2, self.num_channels_latents, condition.shape[1]),
                    device=exec_device,
                    dtype=condition.dtype,
                )
                t_expanded_2 = torch.empty(2, device=exec_device, dtype=condition.dtype)
                t_expanded_1 = torch.empty(1, device=exec_device, dtype=condition.dtype)
                for t in timesteps:
                    if overlap > 0:
                        time_value = t.to(latents.dtype)
                        latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * time_value) * noise_prompt + (
                            time_value * previous_latent[..., :overlap]
                        )
                    is_early = getattr(scheduler, "current_is_early", True)
                    current_cfg = early_cfg if is_early else late_cfg
                    can_compact = (not is_early) and (abs(current_cfg - 1.0) < 1e-5)
                    if can_compact:
                        t_expanded_1.fill_(t)
                        velocity = self.transformer(
                            hidden_states=latents,
                            timestep=t_expanded_1,
                            encoder_hidden_states=condition,
                        )
                    else:
                        t_expanded_2.fill_(t)
                        batch_latents_2[0] = latents[0]
                        batch_latents_2[1] = latents[0]
                        batch_pred = self.transformer(
                            hidden_states=batch_latents_2,
                            timestep=t_expanded_2,
                            encoder_hidden_states=batch_cond,
                        )
                        noise_pred_cond = batch_pred[0:1]
                        noise_pred_uncond = batch_pred[1:2]
                        velocity = noise_pred_uncond + current_cfg * (noise_pred_cond - noise_pred_uncond)
                    latents = scheduler.step(velocity, t, latents, generator=chunk_gen)
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
                del noise_prompt, condition, uncond_condition, batch_cond, batch_latents_2, t_expanded_2, t_expanded_1
        finally:
            if pbar is not None:
                pbar.close()
            del previous_latent, previous_condition
            if exec_device.type == "cuda":
                torch.cuda.empty_cache()
        return latent_chunks

    @torch.no_grad()
    def decode_latents(
        self,
        latent_chunks: List[torch.Tensor],
        batch_size: int = 4,
        show_progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        num_chunks = len(latent_chunks)
        if num_chunks == 0:
            return torch.empty((1, 2, 0), dtype=torch.float32)
        vocoder_dtype = self.vocoder.dec_in_proj.weight.dtype
        waveform_chunks: List[torch.Tensor] = []
        chunk_idx = 0
        total_batches = math.ceil(num_chunks / max(1, batch_size))
        pbar = None
        if show_progress and progress_callback is None:
            pbar = tqdm(
                total=total_batches,
                desc="Stage 3 [Flow-VAE Vocoder]",
                dynamic_ncols=True,
                unit="batch",
            )
        batch_counter = 0
        seam_len = _SEAM_OVERLAP_SAMPLES
        theta = torch.linspace(0.0, math.pi / 2.0, seam_len, dtype=torch.float32)
        w_out = torch.cos(theta)
        w_in = torch.sin(theta)
        pending_tail: Optional[torch.Tensor] = None
        try:
            while chunk_idx < num_chunks:
                target_shape = latent_chunks[chunk_idx].shape
                batch_end = chunk_idx + 1
                while (
                    batch_end < num_chunks
                    and (batch_end - chunk_idx) < batch_size
                    and latent_chunks[batch_end].shape == target_shape
                ):
                    batch_end += 1
                batch_slice = [latent_chunks[k] for k in range(chunk_idx, batch_end)]
                batch_latents = torch.cat(batch_slice, dim=0).to(dtype=vocoder_dtype)
                batch_waveforms = self.vocoder(batch_latents)
                for local_idx, global_idx in enumerate(range(chunk_idx, batch_end)):
                    wv = batch_waveforms[local_idx : local_idx + 1]
                    wv_len = wv.shape[-1]
                    left_crop = 0 if global_idx == 0 else _CROP_LEFT_LATENT * self.latent_hop_length
                    right_crop = 0 if global_idx == num_chunks - 1 else _CROP_RIGHT_LATENT * self.latent_hop_length
                    end_idx = wv_len if right_crop == 0 else (wv_len - right_crop)
                    if num_chunks == 1:
                        waveform_chunks.append(wv.cpu())
                    elif global_idx == 0:
                        waveform_chunks.append(wv[..., : end_idx - seam_len].cpu())
                        pending_tail = wv[..., end_idx - seam_len : end_idx].cpu()
                    elif global_idx == num_chunks - 1:
                        head = wv[..., left_crop : left_crop + seam_len].cpu()
                        if pending_tail is not None and pending_tail.shape[-1] == seam_len and head.shape[-1] == seam_len:
                            blended = pending_tail * w_out + head * w_in
                            waveform_chunks.append(blended)
                        waveform_chunks.append(wv[..., left_crop + seam_len :].cpu())
                        pending_tail = None
                    else:
                        head = wv[..., left_crop : left_crop + seam_len].cpu()
                        if pending_tail is not None and pending_tail.shape[-1] == seam_len and head.shape[-1] == seam_len:
                            blended = pending_tail * w_out + head * w_in
                            waveform_chunks.append(blended)
                        waveform_chunks.append(wv[..., left_crop + seam_len : end_idx - seam_len].cpu())
                        pending_tail = wv[..., end_idx - seam_len : end_idx].cpu()
                del batch_latents, batch_waveforms
                for k in range(chunk_idx, batch_end):
                    latent_chunks[k] = None
                chunk_idx = batch_end
                batch_counter += 1
                if pbar is not None:
                    pbar.update(1)
                if progress_callback is not None:
                    progress_callback(batch_counter, total_batches)
        finally:
            if pbar is not None:
                pbar.close()
        return torch.cat(waveform_chunks, dim=-1).float()