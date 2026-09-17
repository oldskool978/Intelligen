from __future__ import annotations

import os
import sys
import math
import json
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import argparse
import traceback
import numpy as np

from schema import (
    GenerationRequest,
    GenerationResponse,
    SUPPORTED_SOLVERS,
    get_active_engine_defaults,
    has_custom_default_preset,
    DEFAULT_PRESET_FILENAME,
    parse_k_vector,
)
from engine import MusicEngine

DEFAULT_HARNESS_VOCAL_LYRICS = """[intro]
(Smooth Rhodes chords, filtered 808 glide, ad-libs)
Yeah, listen
Midnight in the city, let the groove breathe
Oh, oh-woah, yeah

[verse 1]
Midnight riding under neon streetlights
Searching for the answers in the rearview mirror
Thought I had the blueprint solid in my mind
Now the silhouette of you is drawing nearer
Dashboard glowing with a steady slow pulse
Echoes of your whisper in the night air

[pre-chorus 1]
I try to fight it, but it's pulling me in
Every harmonic frequency starts spinning again
Tension rising from the bottom to top
Got that momentum and we never gon' stop

[chorus 1]
Got me caught up in the way that you move
Nobody else can lock right into the groove
Got my heart on the floor, baby, give me one more
Show me that rhythm, tell me what you wanna do
(Yeah, yeah, keep it right there)

[verse 2]
Two in the morning, baseline taking over
Sip of something smooth, leaning in a little closer
Sub-frequencies vibrating the floor
You give me everything, but I still want more
Syncopated touch, perfect timing on the beat
Fire in our eyes, generating pure heat

[pre-chorus 2]
I try to fight it, but it's pulling me in
Every harmonic frequency starts spinning again
Tension rising from the bottom to top
Got that momentum and we never gon' stop

[chorus 2]
Got me caught up in the way that you move
Nobody else can lock right into the groove
Got my heart on the floor, baby, give me one more
Show me that rhythm, tell me what you wanna do
(Yeah, yeah, right into the pocket)

[bridge]
Take it to the falsetto high, let the bass drop clean
Smoothest vibration that you've ever seen
Counterpoint melodies weaving around
Elevating the pressure, capturing the sound
Hold that note, let the energy soar
Take it to places that we never went before

[solo]
(Warm expressive nylon and electric guitar soloing over deep sub-bass and syncopated percussion)

[chorus 3]
Got me caught up in the way that you move
Nobody else can lock right into the groove
Got my heart on the floor, baby, give me one more
Show me that rhythm, tell me what you wanna do
(Oh-woah, give me one more time)

[outro]
Fade into the low-end frequency
Keep the drum pocket steady for me
Ad-libs drifting out into the night
Yeah, just like that
Fade to black"""

DEFAULT_HARNESS_INSTRUMENTAL_CUES = """[intro]
(Warm Fender Rhodes chords with subtle vinyl crackle and distant tape delay)

[theme a]
(Sub-bass 808 glide enters alongside crisp syncopated rimshot and closed hi-hats)

[verse 1]
(Acoustic nylon guitar plucks introduce the primary melodic motif over a steady groove)

[pre-chorus 1]
(Rising synth pad swells building dynamic tension with filtered white noise sweep)

[chorus 1]
(Full punchy kick drum drops in, melodic lead synth takes center stage with stereo widening)

[verse 2]
(Percussion strips back slightly, expressive legato electric guitar answers the chord changes)

[pre-chorus 2]
(Tension builds with rolling 32nd-note hi-hat accents and rising brass stabs)

[chorus 2]
(Climactic drop with full rhythmic section, driving 808, and soaring harmonic layers)

[bridge]
(Half-time rhythmic breakdown with filtered Rhodes chords and resonant sub drops)

[solo]
(Virtuosic expressive electric guitar solo with dynamic slides and warm tube overdrive)

[chorus 3]
(Final explosive climax with layered counter-melodies and maximum harmonic punch)

[outro]
(Drums fade gradually, leaving solitary Rhodes chords and decaying reverb tails to silence)"""


def calculate_schedule_partition(
    num_steps: int,
    handoff_threshold: float,
    audio_duration: float,
    early_solver: str,
    late_solver: str,
    late_cfg: float = 1.0000,
    sampling_rate: int = 44100,
) -> Tuple[int, int, int, int, float]:
    base_shift = 0.50
    max_shift = 1.15
    base_seq_len = 256
    max_seq_len = 4096
    latent_seq_len = int(math.ceil((audio_duration * sampling_rate) / 512))
    ratio = max(0.0, min(1.0, (latent_seq_len - base_seq_len) / float(max_seq_len - base_seq_len)))
    dyn_shift = base_shift + ratio * (max_shift - base_shift)

    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps, dtype=np.float64)
    if dyn_shift != 1.0:
        sigmas = dyn_shift * sigmas / (1.0 + (dyn_shift - 1.0) * sigmas)
    sigmas = 1.0 - sigmas

    if handoff_threshold >= 1.0:
        k_star = num_steps
    elif handoff_threshold <= 0.0:
        k_star = 0
    else:
        cand = [i for i in range(num_steps) if sigmas[i] >= handoff_threshold]
        k_star = cand[0] if cand else num_steps

    early_steps = k_star
    late_steps = num_steps - k_star

    corrector_types = ("heun", "sde_gpu_pp", "multitree", "res_multistep", "multires")
    early_has_corrector = early_solver in corrector_types
    late_has_corrector = late_solver in corrector_types

    early_nfe = early_steps * (2 if early_has_corrector else 1)
    late_nfe = late_steps * (2 if late_has_corrector else 1)
    total_nfe = early_nfe + late_nfe

    late_can_compact = abs(late_cfg - 1.0) < 1e-5
    early_evals = early_nfe * 2
    late_evals = late_nfe * (1 if late_can_compact else 2)
    total_evals = early_evals + late_evals

    baseline_evals = (num_steps * 2) * 2
    reduction_pct = max(0.0, (1.0 - (total_evals / float(baseline_evals))) * 100.0)

    return early_steps, late_steps, total_nfe, total_evals, reduction_pct


def create_default_harness_request() -> GenerationRequest:
    defaults = get_active_engine_defaults()
    default_vocal = "Silky male tenor lead vocal, dynamic chest-to-falsetto transitions, intricate melismatic ad-libs, stacked 4-part harmonies."
    default_inst = "Warm Fender Rhodes chords, expressive legato nylon guitar, and melodic synthesizer leads."

    return GenerationRequest(
        genre="Contemporary R&B",
        subgenre="2000s Pop R&B / Slow Jam Bounce",
        bpm=96,
        key="F minor",
        mood="Sensual, passionate, smooth, confident, driving.",
        vocals=default_vocal,
        vocal_lead=default_vocal,
        instrumental_lead=default_inst,
        arrangement="Deep 808 sub-bass, crisp acoustic-electronic hybrid snare on 2 and 4, syncopated hi-hat rolls, warm Fender Rhodes chords.",
        lyrics=DEFAULT_HARNESS_VOCAL_LYRICS,
        instrumental_lyrics=DEFAULT_HARNESS_INSTRUMENTAL_CUES,
        is_instrumental=False,
        instrumental_branch="cues",
        temperature=defaults["temperature"],
        top_p=defaults["top_p"],
        top_k=defaults["top_k"],
        top_k_layers=list(defaults["top_k_layers"]),
        ar_guidance_scale=defaults["ar_guidance_scale"],
        early_instrumental_solver=defaults["early_instrumental_solver"],
        late_instrumental_solver=defaults["late_instrumental_solver"],
        early_vocal_solver=defaults["early_vocal_solver"],
        late_vocal_solver=defaults["late_vocal_solver"],
        handoff_threshold=defaults["handoff_threshold"],
        num_inference_steps=defaults["num_inference_steps"],
        instrumental_guidance_scale=defaults["instrumental_guidance_scale"],
        early_instrumental_cfg=defaults["early_instrumental_cfg"],
        late_instrumental_cfg=defaults["late_instrumental_cfg"],
        vocal_guidance_scale=defaults["vocal_guidance_scale"],
        early_vocal_cfg=defaults["early_vocal_cfg"],
        late_vocal_cfg=defaults["late_vocal_cfg"],
        eta=defaults["eta"],
        s_noise=defaults["s_noise"],
        vocoder_batch_size=defaults.get("vocoder_batch_size", 2),
        audio_duration=300.0,
        seed=42,
        output_path="output_vocal_master.wav",
        apply_declick=defaults["apply_declick"],
        cpu_offload=defaults["cpu_offload"],
    )


def format_k_vector_display(k_list: List[int]) -> str:
    labels = ["L0:Sem", "L1:Tim0", "L2:Tim1", "L3:Tim2", "L4:Phs0", "L5:Phs1", "L6:Phs2", "L7:Phs3"]
    return " | ".join([f"{lbl}={k_list[i]}" for i, lbl in enumerate(labels)])


def print_telemetry(resp: GenerationResponse, req: Optional[GenerationRequest] = None) -> None:
    print("\n" + "=" * 84)
    print("                        ACOUSTIC TELEMETRY REPORT")
    print("=" * 84)
    print(f"Master Destination:    {resp.output_path}")
    print(f"Sampling Resolution:   {resp.sample_rate} Hz (32-bit Float PCM)")
    print(f"Audio Duration:        {resp.duration_seconds:.4f}s ({resp.total_samples:,} samples)")
    print(f"Inference Latency:     {resp.generation_time_seconds:.4f}s (RTF: {resp.real_time_factor:.4f}x)")
    print(f"Peak VRAM Footprint:   {resp.peak_vram_gb:.3f} GB")
    print(f"Memory Architecture:   {'SEQUENTIAL CPU OFFLOAD' if resp.cpu_offload_active else 'RESIDENT GPU VRAM'}")
    print(f"Modality Mode:         {'INSTRUMENTAL' if resp.is_instrumental_used else 'VOCAL SONG'}")
    if resp.is_instrumental_used:
        print(f"Instrumental Branch:   {resp.instrumental_branch_used.upper()}")
    if req is not None:
        print(f"Vocoder Batch Size:    {req.vocoder_batch_size} (Vectorized Waveform Parallelism)")
    print(f"Depth K-Search Vector: {resp.top_k_vector_used}")
    print(f"Flow Handoff Boundary: t* = {resp.handoff_threshold_used:.4f} (Early: {resp.early_steps} steps | Late: {resp.late_steps} steps)")
    print(f"Evaluated Passes (NFE):{resp.total_nfe_chunk} forward evaluations per chunk")

    early_solv = resp.early_instrumental_solver_used if resp.is_instrumental_used else resp.early_vocal_solver_used
    late_solv = resp.late_instrumental_solver_used if resp.is_instrumental_used else resp.late_vocal_solver_used
    print(f"ODE Solvers:           Early: {early_solv.upper()} | Late: {late_solv.upper()}")

    early_cfg = resp.early_instrumental_cfg_used if resp.is_instrumental_used else resp.early_vocal_cfg_used
    late_cfg = resp.late_instrumental_cfg_used if resp.is_instrumental_used else resp.late_vocal_cfg_used
    print(f"Flow-Match Guidance:   Early CFG: {early_cfg:.4f} | Late CFG: {late_cfg:.4f}")

    if resp.eta_used > 0.0:
        print(f"Dispersion Field:      Eta: {resp.eta_used:.4f} | S-Noise: {resp.s_noise_used:.4f}")
    print(f"Boundary Conditioning: {'SYMMETRIC SUB-MS HANN DE-CLICK' if resp.declick_applied else 'BYPASS RAW SAMPLES'}")
    print(f"Signal Dynamics (Peak):{resp.peak_linear:.8f} ({resp.peak_dbfs:.4f} dBFS)")
    print(f"Signal Dynamics (RMS): {resp.rms_dbfs:.4f} dBFS")
    print(f"Acoustic Crest Factor: {resp.crest_factor_db:.4f} dB")
    print("-" * 84)
    print(f"Effective Conditioning Prompt:\n{resp.effective_prompt}")
    print("=" * 84 + "\n")


def display_menu(req: GenerationRequest, engine: Optional[MusicEngine] = None) -> None:
    defaults = get_active_engine_defaults()
    t_disp = f"{req.temperature:.4f}" if req.temperature is not None else f"{defaults['temperature']:.4f}"
    p_disp = f"{req.top_p:.4f}" if req.top_p is not None else f"{defaults['top_p']:.4f}"
    ar_cfg_disp = f"{req.ar_guidance_scale:.4f}" if req.ar_guidance_scale is not None else f"{defaults['ar_guidance_scale']:.4f}"
    steps_val = req.num_inference_steps if req.num_inference_steps is not None else defaults["num_inference_steps"]
    steps_disp = str(steps_val)

    is_inst = req.is_instrumental
    e_solv = req.early_instrumental_solver if is_inst else req.early_vocal_solver
    l_solv = req.late_instrumental_solver if is_inst else req.late_vocal_solver
    e_cfg = req.early_instrumental_cfg if is_inst else req.early_vocal_cfg
    l_cfg = req.late_instrumental_cfg if is_inst else req.late_vocal_cfg

    ei_cfg_disp = f"{e_cfg:.4f}" if e_cfg is not None else f"{defaults['early_instrumental_cfg']:.4f}"
    li_cfg_disp = f"{l_cfg:.4f}" if l_cfg is not None else f"{defaults['late_instrumental_cfg']:.4f}"
    eta_disp = f"{req.eta:.4f}" if req.eta is not None else f"{defaults['eta']:.4f}"
    s_noise_disp = f"{req.s_noise:.4f}" if req.s_noise is not None else f"{defaults['s_noise']:.4f}"
    declick_disp = "ENABLED (Symmetric Hann)" if req.apply_declick else "DISABLED"
    offload_disp = "ENABLED (Sequential Streaming)" if req.cpu_offload else "DISABLED (Resident VRAM)"

    k_vec = req.resolve_top_k_layers()
    k_vec_str = format_k_vector_display(k_vec)

    active_lyrics = req.instrumental_lyrics if (is_inst and req.instrumental_branch == "cues") else req.lyrics
    lyrics_status = f"{len(active_lyrics.splitlines())} lines configured" if active_lyrics.strip() else "<Empty Sheet>"
    anchor_tag = "Intelligen/default.json (Active File)" if has_custom_default_preset() else "Discovered Optimal Baseline (Hardcoded)"

    early_steps, late_steps, total_nfe, total_evals, red_pct = calculate_schedule_partition(
        num_steps=steps_val,
        handoff_threshold=req.handoff_threshold,
        audio_duration=req.audio_duration,
        early_solver=e_solv,
        late_solver=l_solv,
        late_cfg=l_cfg if l_cfg is not None else 1.0,
    )

    lead_header = "Acoustic Lead:" if is_inst else "Vocal Profile:"
    lead_content = (req.instrumental_lead or req.vocals) if is_inst else (req.vocal_lead or req.vocals)

    print("\n" + "=" * 84)
    print("               MINIMAX-MUSIC3 MODALITY EXPLORATION & ABLATION HARNESS")
    print(f"                       [{anchor_tag}]")
    print("=" * 84)
    print(" --- PRODUCTION BRIEF (PERSISTENT SONG DRAFT) ---")
    print(f" [M]  Active Modality:       {'INSTRUMENTAL' if is_inst else 'VOCAL SONG'} (Branch: {req.instrumental_branch.upper() if is_inst else 'SONG MASTER'})")
    print(f" [1]  Genre & Subgenre:      {req.genre} / {req.subgenre}")
    print(f" [2]  BPM:                   {req.bpm}")
    print(f" [3]  Key Signature:         {req.key}")
    print(f" [4]  Mood Narrative:        {req.mood}")
    print(f" [5]  {lead_header:<22} {lead_content}")
    print(f" [6]  Arrangement Details:   {req.arrangement}")
    print(f" [7]  Raw Prompt Override:   {req.raw_prompt if req.raw_prompt else '<Auto-Compiled 3-Heading Hierarchy>'}")
    print(f" [E]  Edit Active Sheet:     {lyrics_status} ({'Instrumental Cues' if (is_inst and req.instrumental_branch == 'cues') else 'Vocal Lyrics'})")
    print(" --- STAGE 1 AUTOREGRESSIVE GENERATION ---")
    print(f" [8]  Temperature & AR CFG:  T: {t_disp} | AR CFG: {ar_cfg_disp}")
    print(f" [9]  Nucleus Top-P:         Top-P: {p_disp}")
    print(f" [10] Hierarchical K-Vector: [ {k_vec_str} ]")
    print(" --- STAGE 2 FLOW-MATCHING (TWO-REGIME MULTI-RATE ODE) ---")
    print(f" [11] Early Solver:          {e_solv.upper()}")
    print(f" [12] Late Solver:           {l_solv.upper()}")
    print(f" [13] Handoff Threshold:     t* = {req.handoff_threshold:.4f} (Early: {early_steps} st | Late: {late_steps} st | Evals: {total_evals}/chk, -{red_pct:.1f}%)")
    print(f" [14] Inference Steps:       {steps_disp}")
    print(f" [15] CFG Field:             Early CFG: {ei_cfg_disp} | Late CFG: {li_cfg_disp}")
    print(f" [16] SDE Dispersion (Early):Eta: {eta_disp} | S-Noise: {s_noise_disp}")
    print(" --- STAGE 3 VOCODER DECODING & HARDWARE ---")
    print(f" [17] Track Length Ceiling:  {req.audio_duration:.4f}s")
    print(f" [18] PRNG Generation Seed:  {req.seed}")
    print(f" [19] Output WAV Path:       {req.output_path}")
    print(f" [21] DSP Boundary De-Click: {declick_disp}")
    print(f" [22] Memory CPU Streaming:  {offload_disp}")
    print(f" [23] Vocoder Batch Size:    {req.vocoder_batch_size} (Power-of-Two Parallelism: 1, 2, 4)")
    print("-" * 84)
    print(" --- EXECUTION & INSTRUMENTAL EXPERIMENTATION ---")
    print(f" [G]  Generate Master Track ({'INSTRUMENTAL: ' + req.instrumental_branch.upper() if is_inst else 'VOCAL SONG MASTER'})")
    print(f" [I]  Configure Instrumental Mode (Branch 1: Bare Tags | Branch 2: Cues)")
    print(f" [T1] Run Bare-Tag Instrumental (Song tags preserved, lyrics stripped downwind)")
    print(f" [T2] Run Arrangement-Cue Instrumental (Dedicated parenthetical directives)")
    print(f" [P]  Preview 3-Heading Prompt & Downwind Token Sequence")
    print(f" [T]  Reset Baseline    [L] Load Preset    [S] Save Preset    [D] Direct Save default.json")
    print(f" [Q]  Quit Harness")
    print("=" * 84)


def edit_k_topology_submenu(req: GenerationRequest) -> None:
    while True:
        k_vec = req.resolve_top_k_layers()
        print("\n" + "-" * 76)
        print("          HIERARCHICAL RVQ LAYER-WISE CANDIDATE TOPOLOGY (K-VECTOR)")
        print("-" * 76)
        print(f" Current Vector: [ {format_k_vector_display(k_vec)} ]")
        print("\n [M] Set Macro Bands (Fundamental: L0 | Acoustic: L1-3 | Fine: L4-7)")
        print(" [0] Layer 0 (Global LM Semantic)   : ", k_vec[0])
        print(" [1] Layer 1 (RVQ Depth Head 0)     : ", k_vec[1])
        print(" [2] Layer 2 (RVQ Depth Head 1)     : ", k_vec[2])
        print(" [3] Layer 3 (RVQ Depth Head 2)     : ", k_vec[3])
        print(" [4] Layer 4 (RVQ Depth Head 3)     : ", k_vec[4])
        print(" [5] Layer 5 (RVQ Depth Head 4)     : ", k_vec[5])
        print(" [6] Layer 6 (RVQ Depth Head 5)     : ", k_vec[6])
        print(" [7] Layer 7 (RVQ Depth Head 6)     : ", k_vec[7])
        print(" [V] Enter Full 8-Element Vector (e.g. 47,47,47,45,39,37,38,39)")
        print(" [D] Reset to Discovered Optimal Vector [47, 47, 47, 45, 39, 37, 38, 39]")
        print(" [R] Reset to Standard Flat Baseline (All 47)")
        print(" [B] Back to Main Harness")
        print("-" * 76)
        sub_choice = input("Select operation: ").strip().upper()
        if sub_choice in ("B", ""):
            break
        elif sub_choice == "D":
            req.top_k_layers = [47, 47, 47, 45, 39, 37, 38, 39]
            req.top_k = 47
            print("K-search vector reset to discovered optimal [47, 47, 47, 45, 39, 37, 38, 39].")
        elif sub_choice == "R":
            req.top_k_layers = [47, 47, 47, 47, 47, 47, 47, 47]
            req.top_k = 47
            print("K-search vector reset to flat baseline [47, 47, 47, 47, 47, 47, 47, 47].")
        elif sub_choice == "M":
            f_in = input(f"Fundamental Band K (Layer 0) [{k_vec[0]}]: ").strip()
            a_in = input(f"Acoustic Band K (Layers 1..3) [{k_vec[1]}]: ").strip()
            fn_in = input(f"Fine Band K (Layers 4..7) [{k_vec[4]}]: ").strip()
            f_val = int(f_in) if f_in.isdigit() else k_vec[0]
            a_val = int(a_in) if a_in.isdigit() else k_vec[1]
            fn_val = int(fn_in) if fn_in.isdigit() else k_vec[4]
            req.set_macro_k(f_val, a_val, fn_val)
            print(f"Macro bands updated: Fundamental={f_val}, Acoustic={a_val}, Fine={fn_val}")
        elif sub_choice in ["0", "1", "2", "3", "4", "5", "6", "7"]:
            idx = int(sub_choice)
            val_in = input(f"Enter candidate count K for Layer {idx} (1-500) [{k_vec[idx]}]: ").strip()
            if val_in.isdigit() and 1 <= int(val_in) <= 500:
                req.set_layer_k(idx, int(val_in))
        elif sub_choice == "V":
            raw_v = input("Enter comma-separated 8 integers: ").strip()
            parsed = parse_k_vector(raw_v)
            if parsed:
                req.top_k_layers = parsed
                req.top_k = parsed[0]
                print(f"Updated K-search vector to {parsed}")
            else:
                print("Error: Expected exactly 8 integer values.")


def edit_multiline_sheet(current_text: str, is_inst: bool) -> str:
    print(f"\n--- Edit {'Instrumental Directives' if is_inst else 'Vocal Lyrics'} ---")
    if current_text.strip():
        print(current_text)
    else:
        print("<Currently Empty>")
    print("\nEnter content (Type '__DONE__' on an empty line to finish, or '__CLEAR__' to erase):")
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "__DONE__":
                break
            if line.strip() == "__CLEAR__":
                return ""
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines).strip()


def prompt_solver_selection(prompt_label: str, current_val: str) -> str:
    sub_map = {
        "1": "heun",
        "heun": "heun",
        "2": "euler",
        "euler": "euler",
        "3": "ipndm",
        "ipndm": "ipndm",
        "4": "sde_gpu_pp",
        "sde": "sde_gpu_pp",
        "sde_gpu_pp": "sde_gpu_pp",
        "5": "multitree",
        "multitree": "multitree",
        "multires": "multitree",
        "res_multistep": "multitree",
    }
    print(f"\nSelect {prompt_label}: [1] HEUN  [2] EULER  [3] IPNDM  [4] SDE_GPU_PP  [5] MULTITREE")
    sel = input(f"Choice [{current_val}]: ").strip().lower()
    return sub_map.get(sel, current_val)


def prompt_instrumental_menu(req: GenerationRequest) -> None:
    print("\n" + "-" * 76)
    print("                     INSTRUMENTAL MODE CONFIGURATION")
    print("-" * 76)
    print(f" Current Modality: {'INSTRUMENTAL' if req.is_instrumental else 'VOCAL SONG'}")
    if req.is_instrumental:
        print(f" Active Branch:   {req.instrumental_branch.upper()}")
    print("\n [0] Return to Vocal Song Mode (Full singing voice & lyrics)")
    print(" [1] Branch A: Bare-Tag Projection (Song tags preserved, words stripped downwind)")
    print(" [2] Branch B: Arrangement Directives (Parenthetical cues for pacing/dynamics)")
    print("-" * 76)
    c = input("Select branch [0-2]: ").strip()
    if c == "0":
        req.is_instrumental = False
        req.vocals = req.vocal_lead or req.vocals
        req.output_path = "output_vocal_master.wav"
        print("\nModality set to: VOCAL SONG")
    elif c == "1":
        req.is_instrumental = True
        req.instrumental_branch = "tags_only"
        req.vocals = req.instrumental_lead or ""
        req.output_path = "output_bare_tags.wav"
        print("\nModality set to: INSTRUMENTAL (Branch A: Bare Tags)")
    elif c == "2":
        req.is_instrumental = True
        req.instrumental_branch = "cues"
        req.vocals = req.instrumental_lead or ""
        req.output_path = "output_arrangement_cues.wav"
        if not req.instrumental_lyrics.strip():
            req.instrumental_lyrics = DEFAULT_HARNESS_INSTRUMENTAL_CUES
        print("\nModality set to: INSTRUMENTAL (Branch B: Arrangement Cues)")


def run_interactive_harness(engine: Optional[MusicEngine], initial_req: Optional[GenerationRequest] = None) -> None:
    req = initial_req if initial_req is not None else create_default_harness_request()
    defaults = get_active_engine_defaults()

    while True:
        display_menu(req, engine)
        choice = input("Select action or field: ").strip().upper()

        if choice == "M":
            req.is_instrumental = not req.is_instrumental
            if req.is_instrumental:
                req.vocals = req.instrumental_lead or ""
                req.output_path = "output_instrumental.wav"
            else:
                req.vocals = req.vocal_lead or req.vocals
                req.output_path = "output_vocal_master.wav"
            print(f"\nSwitched modality to: {'INSTRUMENTAL' if req.is_instrumental else 'VOCAL SONG'}")

        elif choice == "I":
            prompt_instrumental_menu(req)

        elif choice == "1":
            g = input(f"Enter Genre [{req.genre}]: ").strip()
            if g:
                req.genre = g
            sg = input(f"Enter Subgenre [{req.subgenre}]: ").strip()
            if sg:
                req.subgenre = sg

        elif choice == "2":
            b = input(f"Enter BPM (30 - 300, 0 for unmetered) [{req.bpm}]: ").strip()
            if b.isdigit() and (int(b) == 0 or 30 <= int(b) <= 300):
                req.bpm = int(b)

        elif choice == "3":
            k = input(f"Enter Key Signature [{req.key}]: ").strip()
            if k:
                req.key = k

        elif choice == "4":
            m = input(f"Enter Mood Narrative [{req.mood}]: ").strip()
            if m:
                req.mood = m

        elif choice == "5":
            if req.is_instrumental:
                curr = req.instrumental_lead or req.vocals
                v = input(f"Enter Instrumental Lead / Acoustic Character [{curr}]: ").strip()
                if v:
                    req.instrumental_lead = v
                    req.vocals = v
            else:
                curr = req.vocal_lead or req.vocals
                v = input(f"Enter Vocal Profile & Character [{curr}]: ").strip()
                if v:
                    req.vocal_lead = v
                    req.vocals = v

        elif choice == "6":
            a = input(f"Enter Arrangement Details [{req.arrangement}]: ").strip()
            if a:
                req.arrangement = a

        elif choice == "7":
            r = input("Enter Raw Prompt override (empty to reset to 3-heading compositor): ").strip()
            req.raw_prompt = r if r else None

        elif choice == "E":
            if req.is_instrumental and req.instrumental_branch == "cues":
                req.instrumental_lyrics = edit_multiline_sheet(req.instrumental_lyrics, is_inst=True)
            else:
                req.lyrics = edit_multiline_sheet(req.lyrics, is_inst=False)

        elif choice == "8":
            t = input(
                f"Enter Sampling Temperature [{req.temperature if req.temperature is not None else defaults['temperature']}]: "
            ).strip()
            req.temperature = float(t) if t and t.lower() != "native" else None
            ar_g = input(
                f"Enter Stage 1 AR Guidance Scale (CFG) [{req.ar_guidance_scale if req.ar_guidance_scale is not None else defaults['ar_guidance_scale']}]: "
            ).strip()
            req.ar_guidance_scale = float(ar_g) if ar_g and ar_g.lower() != "native" else None

        elif choice == "9":
            p = input(f"Enter Top-P [{req.top_p if req.top_p is not None else defaults['top_p']}]: ").strip()
            req.top_p = float(p) if p and p.lower() != "native" else None

        elif choice == "10":
            edit_k_topology_submenu(req)

        elif choice == "11":
            chosen = prompt_solver_selection("Early Solver", req.early_instrumental_solver if req.is_instrumental else req.early_vocal_solver)
            req.early_instrumental_solver = chosen
            req.early_vocal_solver = chosen

        elif choice == "12":
            chosen = prompt_solver_selection("Late Solver", req.late_instrumental_solver if req.is_instrumental else req.late_vocal_solver)
            req.late_instrumental_solver = chosen
            req.late_vocal_solver = chosen

        elif choice == "13":
            h_val = input(f"Enter Flow Handoff Threshold t* [0.0 - 1.0] [{req.handoff_threshold:.4f}]: ").strip()
            if h_val:
                try:
                    req.handoff_threshold = max(0.0, min(1.0, float(h_val)))
                except ValueError:
                    pass

        elif choice == "14":
            s = input(
                f"Enter Steps [{req.num_inference_steps if req.num_inference_steps is not None else defaults['num_inference_steps']}]: "
            ).strip()
            req.num_inference_steps = int(s) if s and s.lower() != "native" else None

        elif choice == "15":
            if req.is_instrumental:
                ei_curr = req.early_instrumental_cfg if req.early_instrumental_cfg is not None else defaults["early_instrumental_cfg"]
                li_curr = req.late_instrumental_cfg if req.late_instrumental_cfg is not None else defaults["late_instrumental_cfg"]
                ei_in = input(f"Enter Early Instrumental CFG [{ei_curr:.4f}]: ").strip()
                if ei_in:
                    req.early_instrumental_cfg = float(ei_in)
                    req.instrumental_guidance_scale = float(ei_in)
                li_in = input(f"Enter Late Instrumental CFG [{li_curr:.4f}]: ").strip()
                if li_in:
                    req.late_instrumental_cfg = float(li_in)
            else:
                ev_curr = req.early_vocal_cfg if req.early_vocal_cfg is not None else defaults["early_vocal_cfg"]
                lv_curr = req.late_vocal_cfg if req.late_vocal_cfg is not None else defaults["late_vocal_cfg"]
                ev_in = input(f"Enter Early Vocal CFG [{ev_curr:.4f}]: ").strip()
                if ev_in:
                    req.early_vocal_cfg = float(ev_in)
                    req.vocal_guidance_scale = float(ev_in)
                lv_in = input(f"Enter Late Vocal CFG [{lv_curr:.4f}]: ").strip()
                if lv_in:
                    req.late_vocal_cfg = float(lv_in)

        elif choice == "16":
            e_curr = req.eta if req.eta is not None else defaults["eta"]
            e_val = input(f"Enter Early SDE / Tree Eta [0.0 - 1.0] [{e_curr:.4f}]: ").strip()
            if e_val:
                req.eta = float(e_val)
            sn_curr = req.s_noise if req.s_noise is not None else defaults["s_noise"]
            sn_val = input(f"Enter Early SDE / Tree S-Noise [0.0 - 5.0] [{sn_curr:.4f}]: ").strip()
            if sn_val:
                req.s_noise = float(sn_val)

        elif choice == "17":
            d = input(f"Enter Duration Ceiling (s) [{req.audio_duration:.4f}]: ").strip()
            if d:
                req.audio_duration = float(d)

        elif choice == "18":
            sd = input(f"Enter PRNG Seed [{req.seed}]: ").strip()
            if sd.isdigit():
                req.seed = int(sd)

        elif choice == "19":
            dst = input(f"Enter Output WAV Path [{req.output_path}]: ").strip()
            if dst:
                req.output_path = dst

        elif choice == "21":
            req.apply_declick = not req.apply_declick

        elif choice == "22":
            req.cpu_offload = not req.cpu_offload

        elif choice == "23":
            print("\nSelect Vocoder Batch Size (Powers of two prevent GPU buffer pool thrashing):")
            print(" [1] Batch Size 1 (Safest VRAM footprint, zero risk of paging)")
            print(" [2] Batch Size 2 (Optimal balanced throughput, standard baseline)")
            print(" [4] Batch Size 4 (Requires >= 16 GB dedicated VRAM headroom)")
            b_in = input(f"Choice [{req.vocoder_batch_size}]: ").strip()
            if b_in.isdigit():
                val = int(b_in)
                if val in (1, 2, 4, 8):
                    req.vocoder_batch_size = val
                    print(f"Vocoder batch size set to {val}.")
                else:
                    print("Recommended: select 1, 2, or 4 to align with CUDA allocator bins.")

        elif choice == "P":
            branch_label = req.instrumental_branch.upper() if req.is_instrumental else "VOCAL SONG"
            print(f"\n--- 3-Heading Composited Prompt ---\n{req.compile_prompt()}\n")
            print(f"--- Sanitized Sequence (Branch: {branch_label}) ---\n{req.sanitize_lyrics()}\n")
            input("Press Enter to continue...")

        elif choice == "T1":
            if engine is None:
                print("\nInitializing neural engine...")
                engine = MusicEngine(repo_id=req.repo_id, device=req.device)
            test_req = req.model_copy(deep=True)
            test_req.is_instrumental = True
            test_req.instrumental_branch = "tags_only"
            test_req.vocals = test_req.instrumental_lead or ""
            test_req.output_path = "output_bare_tags.wav"
            print("\nExecuting Branch A: Bare-Tag Projection...")
            try:
                resp = engine.synthesize(test_req)
                print_telemetry(resp, test_req)
            except Exception as e:
                print(f"Branch A failed: {e}", file=sys.stderr)

        elif choice == "T2":
            if engine is None:
                print("\nInitializing neural engine...")
                engine = MusicEngine(repo_id=req.repo_id, device=req.device)
            test_req = req.model_copy(deep=True)
            test_req.is_instrumental = True
            test_req.instrumental_branch = "cues"
            test_req.vocals = test_req.instrumental_lead or ""
            test_req.output_path = "output_arrangement_cues.wav"
            if not test_req.instrumental_lyrics.strip():
                test_req.instrumental_lyrics = DEFAULT_HARNESS_INSTRUMENTAL_CUES
            print("\nExecuting Branch B: Arrangement Directives...")
            try:
                resp = engine.synthesize(test_req)
                print_telemetry(resp, test_req)
            except Exception as e:
                print(f"Branch B failed: {e}", file=sys.stderr)

        elif choice == "T":
            default_fixture = create_default_harness_request()
            req = default_fixture
            print("\nReset active configuration to discovered baseline.")

        elif choice == "L":
            p_path = input("Enter JSON preset to load: ").strip()
            if not p_path:
                continue
            chosen_path = Path(p_path)
            if not chosen_path.is_absolute() and not chosen_path.exists():
                candidate_preset = ROOT_DIR / "presets" / p_path
                if candidate_preset.exists():
                    chosen_path = candidate_preset
                else:
                    candidate_root = ROOT_DIR / p_path
                    if candidate_root.exists():
                        chosen_path = candidate_root
            try:
                req = GenerationRequest.load_preset(chosen_path)
                print(f"Preset loaded successfully from {chosen_path}")
            except Exception as e:
                print(f"Preset load error: {e}")

        elif choice == "S":
            p_path = input(f"Enter destination JSON preset path (e.g., {DEFAULT_PRESET_FILENAME}): ").strip()
            if not p_path:
                continue
            if not p_path.endswith(".json"):
                p_path = f"{p_path}.json"
            target = Path(p_path)
            if not target.is_absolute():
                if target.name == DEFAULT_PRESET_FILENAME:
                    target = ROOT_DIR / DEFAULT_PRESET_FILENAME
                else:
                    target = ROOT_DIR / "presets" / target.name
            try:
                req.save_preset(target)
                print(f"Preset saved successfully to {target}")
            except Exception as e:
                print(f"Preset save error: {e}")

        elif choice == "D":
            target = ROOT_DIR / DEFAULT_PRESET_FILENAME
            try:
                req.save_preset(target)
                print(f"Authoritative default.json updated at {target}")
            except Exception as e:
                print(f"Preset save error: {e}")

        elif choice == "G":
            if engine is None:
                print("\nInitializing neural engine...")
                engine = MusicEngine(repo_id=req.repo_id, device=req.device)
            resolved_ar_cfg = req.ar_guidance_scale if req.ar_guidance_scale is not None else defaults["ar_guidance_scale"]
            k_vec = req.resolve_top_k_layers()
            steps_active = req.num_inference_steps if req.num_inference_steps is not None else defaults["num_inference_steps"]
            mode_tag = f"INSTRUMENTAL ({req.instrumental_branch.upper()})" if req.is_instrumental else "VOCAL MASTER"
            print(
                f"\nSynthesizing ({mode_tag}, Ceiling={req.audio_duration:.4f}s, Steps={steps_active}, AR_CFG={resolved_ar_cfg:.4f}, K_Vec={k_vec}, VocoderBatch={req.vocoder_batch_size})..."
            )
            try:
                resp = engine.synthesize(req)
                print_telemetry(resp, req)
            except Exception as e:
                print(f"Synthesis failed: {e}", file=sys.stderr)

        elif choice == "Q":
            sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Modality Exploration & Ablation Harness for MiniMax-Music3.")
    parser.add_argument("--batch", action="store_true", help="Run non-interactive generation pass.")
    parser.add_argument("--blank", action="store_true", help="Start with unpopulated fields rather than baseline fixture.")
    parser.add_argument("--instrumental", action="store_true", help="Engage instrumental mode.")
    parser.add_argument("--branch", type=str, choices=["tags_only", "cues"], default="cues")
    parser.add_argument("--genre", type=str, default=None)
    parser.add_argument("--bpm", type=int, default=None)
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--mood", type=str, default=None)
    parser.add_argument("--vocals", type=str, default=None)
    parser.add_argument("--vocal_lead", type=str, default=None)
    parser.add_argument("--inst_lead", "--instrumental_lead", dest="instrumental_lead", type=str, default=None)
    parser.add_argument("--arrangement", type=str, default=None)
    parser.add_argument("--raw_prompt", type=str, default=None)
    parser.add_argument("--lyrics", type=str, default=None)
    parser.add_argument("--inst_lyrics", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_k_layers", type=str, default=None)
    parser.add_argument("--k_macro", nargs=3, type=int, default=None, metavar=("FUNDAMENTAL", "ACOUSTIC", "FINE"))
    parser.add_argument("--ar_cfg", dest="ar_guidance_scale", type=float, default=None)
    parser.add_argument("--early_inst_scheduler", type=str, choices=SUPPORTED_SOLVERS, default=None)
    parser.add_argument("--late_inst_scheduler", type=str, choices=SUPPORTED_SOLVERS, default=None)
    parser.add_argument("--early_voc_scheduler", type=str, choices=SUPPORTED_SOLVERS, default=None)
    parser.add_argument("--late_voc_scheduler", type=str, choices=SUPPORTED_SOLVERS, default=None)
    parser.add_argument("--handoff", "--handoff_threshold", dest="handoff_threshold", type=float, default=None)
    parser.add_argument("--inst_scheduler", type=str, choices=SUPPORTED_SOLVERS, default=None)
    parser.add_argument("--voc_scheduler", type=str, choices=SUPPORTED_SOLVERS, default=None)
    parser.add_argument("--steps", dest="num_inference_steps", type=int, default=None)
    parser.add_argument("--inst_cfg", "--instrumental_guidance_scale", dest="inst_cfg", type=float, default=None)
    parser.add_argument("--early_inst_cfg", type=float, default=None)
    parser.add_argument("--late_inst_cfg", type=float, default=None)
    parser.add_argument("--voc_cfg", "--vocal_guidance_scale", dest="voc_cfg", type=float, default=None)
    parser.add_argument("--early_voc_cfg", type=float, default=None)
    parser.add_argument("--late_voc_cfg", type=float, default=None)
    parser.add_argument("--eta", dest="eta", type=float, default=None)
    parser.add_argument("--s_noise", dest="s_noise", type=float, default=None)
    parser.add_argument("--vocoder_batch_size", dest="vocoder_batch_size", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no_declick", action="store_true", default=False)
    parser.add_argument("--cpu_offload", action="store_true", default=None)
    parser.add_argument("--load_preset", type=str, default=None)
    parser.add_argument("--save_preset", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--repo_id", type=str, default=None)

    args = parser.parse_args()

    if args.load_preset:
        req = GenerationRequest.load_preset(Path(args.load_preset))
    elif args.blank:
        req = GenerationRequest()
    else:
        req = create_default_harness_request()

    if args.instrumental:
        req.is_instrumental = True
    if args.branch is not None:
        req.instrumental_branch = args.branch
    if args.genre is not None:
        req.genre = args.genre
    if args.bpm is not None:
        req.bpm = args.bpm
    if args.key is not None:
        req.key = args.key
    if args.mood is not None:
        req.mood = args.mood
    if args.vocals is not None:
        req.vocals = args.vocals
    if args.vocal_lead is not None:
        req.vocal_lead = args.vocal_lead
    if args.instrumental_lead is not None:
        req.instrumental_lead = args.instrumental_lead
    if args.arrangement is not None:
        req.arrangement = args.arrangement
    if args.raw_prompt is not None:
        req.raw_prompt = args.raw_prompt
    if args.temperature is not None:
        req.temperature = args.temperature
    if args.top_p is not None:
        req.top_p = args.top_p
    if args.top_k is not None:
        req.top_k = args.top_k
        req.top_k_layers = [args.top_k] * 8
    if args.top_k_layers is not None:
        parsed_k = parse_k_vector(args.top_k_layers)
        if parsed_k:
            req.top_k_layers = parsed_k
            req.top_k = parsed_k[0]
    if args.k_macro is not None:
        req.set_macro_k(args.k_macro[0], args.k_macro[1], args.k_macro[2])
    if args.ar_guidance_scale is not None:
        req.ar_guidance_scale = args.ar_guidance_scale
    if args.inst_scheduler is not None:
        req.early_instrumental_solver = args.inst_scheduler
        req.late_instrumental_solver = args.inst_scheduler
    if args.voc_scheduler is not None:
        req.early_vocal_solver = args.voc_scheduler
        req.late_vocal_solver = args.voc_scheduler
    if args.early_inst_scheduler is not None:
        req.early_instrumental_solver = args.early_inst_scheduler
    if args.late_inst_scheduler is not None:
        req.late_instrumental_solver = args.late_inst_scheduler
    if args.early_voc_scheduler is not None:
        req.early_vocal_solver = args.early_voc_scheduler
    if args.late_voc_scheduler is not None:
        req.late_vocal_solver = args.late_voc_scheduler
    if args.handoff_threshold is not None:
        req.handoff_threshold = args.handoff_threshold
    if args.num_inference_steps is not None:
        req.num_inference_steps = args.num_inference_steps
    if args.inst_cfg is not None:
        req.instrumental_guidance_scale = args.inst_cfg
        req.early_instrumental_cfg = args.inst_cfg
    if args.early_inst_cfg is not None:
        req.early_instrumental_cfg = args.early_inst_cfg
        req.instrumental_guidance_scale = args.early_inst_cfg
    if args.late_inst_cfg is not None:
        req.late_instrumental_cfg = args.late_inst_cfg
    if args.voc_cfg is not None:
        req.vocal_guidance_scale = args.voc_cfg
        req.early_vocal_cfg = args.voc_cfg
    if args.early_voc_cfg is not None:
        req.early_vocal_cfg = args.early_voc_cfg
        req.vocal_guidance_scale = args.early_voc_cfg
    if args.late_voc_cfg is not None:
        req.late_vocal_cfg = args.late_voc_cfg
    if args.eta is not None:
        req.eta = args.eta
    if args.s_noise is not None:
        req.s_noise = args.s_noise
    if args.vocoder_batch_size is not None:
        req.vocoder_batch_size = args.vocoder_batch_size
    if args.duration is not None:
        req.audio_duration = args.duration
    if args.seed is not None:
        req.seed = args.seed
    if args.output is not None:
        req.output_path = args.output
    if args.no_declick:
        req.apply_declick = False
    if args.cpu_offload is not None:
        req.cpu_offload = args.cpu_offload
    if args.device is not None:
        req.device = args.device
    if args.repo_id is not None:
        req.repo_id = args.repo_id
    if args.lyrics is not None:
        p = Path(args.lyrics)
        req.lyrics = p.read_text(encoding="utf-8") if p.is_file() else args.lyrics
    if args.inst_lyrics is not None:
        p = Path(args.inst_lyrics)
        req.instrumental_lyrics = p.read_text(encoding="utf-8") if p.is_file() else args.inst_lyrics
    if args.save_preset:
        target = Path(args.save_preset)
        if not target.is_absolute() and target.name == DEFAULT_PRESET_FILENAME:
            target = ROOT_DIR / DEFAULT_PRESET_FILENAME
        req.save_preset(target)
        print(f"Preset exported to {target}")
        sys.exit(0)

    if not args.batch:
        run_interactive_harness(engine=None, initial_req=req)
    else:
        engine = MusicEngine(repo_id=req.repo_id, device=req.device)
        resp = engine.synthesize(req)
        print_telemetry(resp, req)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)