from __future__ import annotations
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import argparse
import traceback
from schema import (
    GenerationRequest,
    GenerationResponse,
    SUPPORTED_SOLVERS,
    SUPPORTED_NOISE_TOPOLOGIES,
    get_active_engine_defaults,
    has_custom_default_preset,
    DEFAULT_PRESET_FILENAME,
    parse_k_vector,
)
from engine import MusicEngine

DEFAULT_HARNESS_LYRICS = """[intro]
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


def create_default_harness_request() -> GenerationRequest:
    defaults = get_active_engine_defaults()
    return GenerationRequest(
        genre="Contemporary R&B",
        subgenre="2000s Pop R&B / Slow Jam Bounce",
        bpm=96,
        key="F minor",
        mood="Sensual, passionate, smooth, confident, driving.",
        vocals="Silky male tenor lead vocal, dynamic chest-to-falsetto transitions, intricate melismatic ad-libs, stacked 4-part harmonies.",
        arrangement="Deep 808 sub-bass, crisp acoustic-electronic hybrid snare on 2 and 4, syncopated hi-hat rolls, warm Fender Rhodes chords.",
        lyrics=DEFAULT_HARNESS_LYRICS,
        temperature=defaults["temperature"],
        top_p=defaults["top_p"],
        top_k=defaults["top_k"],
        top_k_layers=list(defaults["top_k_layers"]),
        ar_guidance_scale=defaults["ar_guidance_scale"],
        instrumental_scheduler=defaults["instrumental_scheduler"],
        vocal_scheduler=defaults["vocal_scheduler"],
        num_inference_steps=defaults["num_inference_steps"],
        instrumental_guidance_scale=defaults["instrumental_guidance_scale"],
        vocal_guidance_scale=defaults["vocal_guidance_scale"],
        eta=defaults["eta"],
        s_noise=defaults["s_noise"],
        noise_topology=defaults["noise_topology"],
        blue_noise_alpha=defaults["blue_noise_alpha"],
        enable_pm_diffusion=defaults["enable_pm_diffusion"],
        pm_iterations=defaults["pm_iterations"],
        pm_conductance=defaults["pm_conductance"],
        pm_lambda=defaults["pm_lambda"],
        audio_duration=300.0,
        seed=42,
        output_path="output.wav",
        apply_declick=defaults["apply_declick"],
        cpu_offload=defaults["cpu_offload"],
    )


def format_k_vector_display(k_list: List[int]) -> str:
    labels = ["L0:Sem", "L1:Tim0", "L2:Tim1", "L3:Tim2", "L4:Phs0", "L5:Phs1", "L6:Phs2", "L7:Phs3"]
    return " | ".join([f"{lbl}={k_list[i]}" for i, lbl in enumerate(labels)])


def print_telemetry(resp: GenerationResponse) -> None:
    print("\n" + "=" * 84)
    print("                        ACOUSTIC TELEMETRY REPORT")
    print("=" * 84)
    print(f"Master Destination:    {resp.output_path}")
    print(f"Sampling Resolution:   {resp.sample_rate} Hz (32-bit Float PCM)")
    print(f"Audio Duration:        {resp.duration_seconds:.4f}s ({resp.total_samples:,} samples)")
    print(f"Inference Latency:     {resp.generation_time_seconds:.4f}s (RTF: {resp.real_time_factor:.4f}x)")
    print(f"Peak VRAM Footprint:   {resp.peak_vram_gb:.3f} GB")
    print(f"Memory Architecture:   {'SEQUENTIAL CPU OFFLOAD' if resp.cpu_offload_active else 'RESIDENT GPU VRAM'}")
    print(f"Depth K-Search Vector: {resp.top_k_vector_used}")
    print(f"ODE Solver Tracks:     INST: {resp.instrumental_scheduler_used.upper()} | VOCAL: {resp.vocal_scheduler_used.upper()}")
    print(f"Bifurcated CFG Field:  Inst Scale: {resp.instrumental_guidance_scale_used:.4f} | Vocal Scale: {resp.vocal_guidance_scale_used:.4f}")
    if resp.instrumental_scheduler_used == "sde_gpu_pp" or resp.vocal_scheduler_used == "sde_gpu_pp":
        print(f"SDE Dispersion:        Eta: {resp.eta_used:.4f} | S-Noise: {resp.s_noise_used:.4f}")
    print(f"Latent Noise Topology: {resp.noise_topology_used.upper()}")
    print(f"Anisotropic PDE (1D):  {'ENABLED (Temporal PM Filter)' if resp.pm_diffusion_used else 'DISABLED'}")
    print(f"Boundary Conditioning: {'SYMMETRIC SUB-MS HANN DE-CLICK' if resp.declick_applied else 'BYPASS RAW SAMPLES'}")
    print(f"Signal Dynamics (Peak):{resp.peak_linear:.8f} ({resp.peak_dbfs:.4f} dBFS)")
    print(f"Signal Dynamics (RMS): {resp.rms_dbfs:.4f} dBFS")
    print(f"Acoustic Crest Factor: {resp.crest_factor_db:.4f} dB")
    print("-" * 84)
    print(f"Effective Conditioning Prompt:\n{resp.effective_prompt}")
    print("=" * 84 + "\n")


def display_menu(req: GenerationRequest) -> None:
    defaults = get_active_engine_defaults()
    t_disp = f"{req.temperature:.4f}" if req.temperature is not None else f"{defaults['temperature']:.4f}"
    p_disp = f"{req.top_p:.4f}" if req.top_p is not None else f"{defaults['top_p']:.4f}"
    ar_cfg_disp = f"{req.ar_guidance_scale:.4f}" if req.ar_guidance_scale is not None else f"{defaults['ar_guidance_scale']:.4f}"
    steps_disp = f"{req.num_inference_steps}" if req.num_inference_steps is not None else f"{defaults['num_inference_steps']}"
    i_cfg_disp = f"{req.instrumental_guidance_scale:.4f}" if req.instrumental_guidance_scale is not None else f"{defaults['instrumental_guidance_scale']:.4f}"
    v_cfg_disp = f"{req.vocal_guidance_scale:.4f}" if req.vocal_guidance_scale is not None else f"{defaults['vocal_guidance_scale']:.4f}"
    eta_disp = f"{req.eta:.4f}" if req.eta is not None else f"{defaults['eta']:.4f}"
    s_noise_disp = f"{req.s_noise:.4f}" if req.s_noise is not None else f"{defaults['s_noise']:.4f}"

    declick_disp = "ENABLED (Symmetric Hann)" if req.apply_declick else "DISABLED"
    offload_disp = "ENABLED (Sequential Streaming)" if req.cpu_offload else "DISABLED (Resident VRAM)"
    pm_disp = (
        f"ENABLED (Iters={req.pm_iterations}, K={req.pm_conductance:.4f}, Lambda={req.pm_lambda:.4f})"
        if req.enable_pm_diffusion
        else "DISABLED"
    )
    k_vec = req.resolve_top_k_layers()
    k_vec_str = format_k_vector_display(k_vec)
    lyrics_status = f"{len(req.lyrics.splitlines())} lines configured" if req.lyrics.strip() else "<Instrumental (Empty)>"
    anchor_tag = "Intelligen/default.json (Active File)" if has_custom_default_preset() else "Discovered Optimal Baseline (Hardcoded)"

    print("\n" + "=" * 84)
    print("               MINIMAX-MUSIC3 MODALITY EXPLORATION & ABLATION HARNESS")
    print(f"                       [{anchor_tag}]")
    print("=" * 84)
    print(" --- AESTHETIC & SEMANTIC CONDITIONING ---")
    print(f" [1]  Genre & Subgenre:      {req.genre} / {req.subgenre}")
    print(f" [2]  BPM:                    {req.bpm}")
    print(f" [3]  Key Signature:          {req.key}")
    print(f" [4]  Mood Narrative:         {req.mood}")
    print(f" [5]  Vocal Architecture:     {req.vocals}")
    print(f" [6]  Arrangement Details:    {req.arrangement}")
    print(f" [7]  Raw Prompt Override:    {req.raw_prompt if req.raw_prompt else '<Auto-Compiled Metadata>'}")
    print(" --- STAGE 1 AUTOREGRESSIVE GENERATION ---")
    print(f" [8]  Temperature & AR CFG:  T: {t_disp} | AR CFG: {ar_cfg_disp}")
    print(f" [9]  Nucleus Top-P:          Top-P: {p_disp}")
    print(f" [10] Hierarchical K-Vector: [ {k_vec_str} ]")
    print(" --- STAGE 2 BIFURCATED FLOW-MATCHING (INSTRUMENTAL & VOCAL ODE) ---")
    print(f" [11] Instrumental Solver:   {req.instrumental_scheduler.upper()}")
    print(f" [12] Vocal Solver:          {req.vocal_scheduler.upper()}")
    print(f" [13] Inference Steps:       {steps_disp}")
    print(f" [14] Bifurcated CFG Field:  Inst: {i_cfg_disp} | Vocal: {v_cfg_disp}")
    print(f" [15] SDE Dispersion:        Eta: {eta_disp} | S-Noise: {s_noise_disp}")
    print(
        f" [16] Latent Prior Topology: {req.noise_topology.upper()}{f' (Alpha: {req.blue_noise_alpha:.4f})' if req.noise_topology == 'blue_noise' else ''}"
    )
    print(f" [17] 1D Temporal PM PDE:    {pm_disp}")
    print(" --- TEMPORAL SYNTHESIS & HARDWARE MEMORY ---")
    print(f" [18] Track Length Ceiling:  {req.audio_duration:.4f}s")
    print(f" [19] PRNG Generation Seed:  {req.seed}")
    print(f" [20] Output WAV Destination:{req.output_path}")
    print(f" [21] Edit Lyrics Payload:   {lyrics_status}")
    print(f" [22] DSP Boundary De-Click: {declick_disp}")
    print(f" [23] Memory CPU Streaming:  {offload_disp}")
    print("-" * 84)
    print(" [P] Print Prompt   [T] Reset Discovered Benchmark   [C] Clear to Instrumental")
    print(" [L] Load Preset    [S] Save Preset (default.json)   [D] Direct Save default.json")
    print(" [G] Generate Audio [Q] Quit")
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


def edit_multiline_lyrics(current_lyrics: str) -> str:
    print("\n--- Edit Structured Lyrics Markup ---")
    if current_lyrics.strip():
        print(current_lyrics)
    else:
        print("<Currently Empty / Pure Instrumental>")
    print("\nEnter new lyrics (Type '__DONE__' on an empty line to finish, or '__CLEAR__' to erase):")
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
    new_text = "\n".join(lines).strip()
    return new_text


def run_interactive_harness(engine: Optional[MusicEngine], req: GenerationRequest) -> None:
    defaults = get_active_engine_defaults()
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
    }
    while True:
        display_menu(req)
        choice = input("Select modality to mutate: ").strip().upper()
        if choice == "1":
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
            v = input(f"Enter Vocal Architecture [{req.vocals}]: ").strip()
            if v:
                req.vocals = v
        elif choice == "6":
            a = input(f"Enter Arrangement Details [{req.arrangement}]: ").strip()
            if a:
                req.arrangement = a
        elif choice == "7":
            r = input("Enter Raw Prompt override (empty to reset): ").strip()
            req.raw_prompt = r if r else None
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
            print("\nSelect Instrumental Solver: [1] HEUN  [2] EULER  [3] IPNDM  [4] SDE_GPU_PP")
            i_sel = input(f"Instrumental Solver [{req.instrumental_scheduler}]: ").strip().lower()
            if i_sel in sub_map:
                req.instrumental_scheduler = sub_map[i_sel]
        elif choice == "12":
            print("\nSelect Vocal Solver: [1] HEUN  [2] EULER  [3] IPNDM  [4] SDE_GPU_PP")
            v_sel = input(f"Vocal Solver [{req.vocal_scheduler}]: ").strip().lower()
            if v_sel in sub_map:
                req.vocal_scheduler = sub_map[v_sel]
        elif choice == "13":
            s = input(
                f"Enter Steps [{req.num_inference_steps if req.num_inference_steps is not None else defaults['num_inference_steps']}]: "
            ).strip()
            req.num_inference_steps = int(s) if s and s.lower() != "native" else None
        elif choice == "14":
            i_curr = req.instrumental_guidance_scale if req.instrumental_guidance_scale is not None else defaults["instrumental_guidance_scale"]
            i_cfg = input(f"Enter Instrumental CFG Scale [{i_curr:.4f}]: ").strip()
            if i_cfg:
                req.instrumental_guidance_scale = float(i_cfg)
            v_curr = req.vocal_guidance_scale if req.vocal_guidance_scale is not None else defaults["vocal_guidance_scale"]
            v_cfg = input(f"Enter Vocal CFG Scale [{v_curr:.4f}]: ").strip()
            if v_cfg:
                req.vocal_guidance_scale = float(v_cfg)
        elif choice == "15":
            e_curr = req.eta if req.eta is not None else defaults["eta"]
            e_val = input(f"Enter SDE Eta [0.0 - 1.0] [{e_curr:.4f}]: ").strip()
            if e_val:
                req.eta = float(e_val)
            sn_curr = req.s_noise if req.s_noise is not None else defaults["s_noise"]
            sn_val = input(f"Enter SDE S-Noise [0.0 - 5.0] [{sn_curr:.4f}]: ").strip()
            if sn_val:
                req.s_noise = float(sn_val)
        elif choice == "16":
            print("\n[1] BLUE_NOISE (High-Pass |f|^alpha)  [2] GAUSSIAN (Standard Normal)")
            n_sel = input(f"Select Noise Topology [{req.noise_topology}]: ").strip()
            if n_sel in ["1", "blue_noise"]:
                req.noise_topology = "blue_noise"
                a_val = input(f"Enter Blue Noise Alpha [0.0 - 2.0] [{req.blue_noise_alpha:.4f}]: ").strip()
                if a_val:
                    req.blue_noise_alpha = float(a_val)
            elif n_sel in ["2", "gaussian"]:
                req.noise_topology = "gaussian"
        elif choice == "17":
            req.enable_pm_diffusion = not req.enable_pm_diffusion
            if req.enable_pm_diffusion:
                i = input(f"PDE Iterations [1-30] [{req.pm_iterations}]: ").strip()
                if i and 1 <= int(i) <= 30:
                    req.pm_iterations = int(i)
                k_val = input(f"PDE Conductance K [0.0001-5.0] [{req.pm_conductance:.4f}]: ").strip()
                if k_val:
                    req.pm_conductance = float(k_val)
                l_val = input(f"PDE Lambda [0.0001-0.25] [{req.pm_lambda:.4f}]: ").strip()
                if l_val:
                    req.pm_lambda = float(l_val)
        elif choice == "18":
            d = input(f"Enter Duration Ceiling (s) [{req.audio_duration:.4f}]: ").strip()
            if d:
                req.audio_duration = float(d)
        elif choice == "19":
            sd = input(f"Enter PRNG Seed [{req.seed}]: ").strip()
            if sd.isdigit():
                req.seed = int(sd)
        elif choice == "20":
            dst = input(f"Enter Output WAV Path [{req.output_path}]: ").strip()
            if dst:
                req.output_path = dst
        elif choice == "21":
            req.lyrics = edit_multiline_lyrics(req.lyrics)
        elif choice == "22":
            req.apply_declick = not req.apply_declick
        elif choice == "23":
            req.cpu_offload = not req.cpu_offload
        elif choice == "P":
            print(f"\n--- Compiled Prompt ---\n{req.compile_prompt()}\n")
            input("Press Enter to continue...")
        elif choice == "T":
            default_fixture = create_default_harness_request()
            req.genre = default_fixture.genre
            req.subgenre = default_fixture.subgenre
            req.bpm = default_fixture.bpm
            req.key = default_fixture.key
            req.mood = default_fixture.mood
            req.vocals = default_fixture.vocals
            req.arrangement = default_fixture.arrangement
            req.lyrics = default_fixture.lyrics
            req.seed = default_fixture.seed
            req.temperature = default_fixture.temperature
            req.top_p = default_fixture.top_p
            req.top_k = default_fixture.top_k
            req.top_k_layers = list(default_fixture.top_k_layers)
            req.ar_guidance_scale = default_fixture.ar_guidance_scale
            req.instrumental_scheduler = default_fixture.instrumental_scheduler
            req.vocal_scheduler = default_fixture.vocal_scheduler
            req.num_inference_steps = default_fixture.num_inference_steps
            req.instrumental_guidance_scale = default_fixture.instrumental_guidance_scale
            req.vocal_guidance_scale = default_fixture.vocal_guidance_scale
            req.eta = default_fixture.eta
            req.s_noise = default_fixture.s_noise
            req.noise_topology = default_fixture.noise_topology
            req.blue_noise_alpha = default_fixture.blue_noise_alpha
            req.enable_pm_diffusion = default_fixture.enable_pm_diffusion
            req.pm_iterations = default_fixture.pm_iterations
            req.pm_conductance = default_fixture.pm_conductance
            req.pm_lambda = default_fixture.pm_lambda
            req.apply_declick = default_fixture.apply_declick
            req.cpu_offload = default_fixture.cpu_offload
            print("\nLoaded complete Midnight Frequency R&B baseline fixture with active defaults.")
        elif choice == "C":
            req.genre = ""
            req.subgenre = ""
            req.bpm = 0
            req.key = ""
            req.mood = ""
            req.vocals = ""
            req.arrangement = ""
            req.lyrics = ""
            req.raw_prompt = None
            req.prompt = None
            print("\nCleared harness conditioning to blank unvoiced instrumental.")
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
                loaded_req = GenerationRequest.load_preset(chosen_path)
                req.genre = loaded_req.genre
                req.subgenre = loaded_req.subgenre
                req.bpm = loaded_req.bpm
                req.key = loaded_req.key
                req.mood = loaded_req.mood
                req.vocals = loaded_req.vocals
                req.arrangement = loaded_req.arrangement
                req.lyrics = loaded_req.lyrics
                req.temperature = loaded_req.temperature
                req.top_p = loaded_req.top_p
                req.top_k = loaded_req.top_k
                req.top_k_layers = list(loaded_req.top_k_layers) if loaded_req.top_k_layers else list(defaults["top_k_layers"])
                req.ar_guidance_scale = loaded_req.ar_guidance_scale
                req.instrumental_scheduler = loaded_req.instrumental_scheduler
                req.vocal_scheduler = loaded_req.vocal_scheduler
                req.num_inference_steps = loaded_req.num_inference_steps
                req.instrumental_guidance_scale = loaded_req.instrumental_guidance_scale
                req.vocal_guidance_scale = loaded_req.vocal_guidance_scale
                req.eta = loaded_req.eta
                req.s_noise = loaded_req.s_noise
                req.noise_topology = loaded_req.noise_topology
                req.blue_noise_alpha = loaded_req.blue_noise_alpha
                req.enable_pm_diffusion = loaded_req.enable_pm_diffusion
                req.pm_iterations = loaded_req.pm_iterations
                req.pm_conductance = loaded_req.pm_conductance
                req.pm_lambda = loaded_req.pm_lambda
                req.apply_declick = loaded_req.apply_declick
                req.cpu_offload = loaded_req.cpu_offload
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

            sched_info = f"Inst: {req.instrumental_scheduler.upper()}, Vocal: {req.vocal_scheduler.upper()}"
            cfg_info = f"Inst_CFG={req.instrumental_guidance_scale:.4f}, Vocal_CFG={req.vocal_guidance_scale:.4f}"

            print(
                f"\nExecuting Synthesis Pass (Ceiling={req.audio_duration:.4f}s, AR_CFG={resolved_ar_cfg:.4f}, K_Vec={k_vec}, {cfg_info}, Solvers=[{sched_info}], Noise={req.noise_topology}, PM={req.enable_pm_diffusion})..."
            )
            try:
                resp = engine.synthesize(req)
                print_telemetry(resp)
            except Exception as e:
                print(f"Synthesis failed: {e}", file=sys.stderr)
        elif choice == "Q":
            sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Modality Exploration & Ablation Harness for MiniMax-Music3.")
    parser.add_argument("--batch", action="store_true", help="Run non-interactive generation pass.")
    parser.add_argument("--blank", action="store_true", help="Start with unpopulated fields rather than R&B fixture.")
    parser.add_argument("--genre", type=str, default=None)
    parser.add_argument("--bpm", type=int, default=None)
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--mood", type=str, default=None)
    parser.add_argument("--vocals", type=str, default=None)
    parser.add_argument("--arrangement", type=str, default=None)
    parser.add_argument("--raw_prompt", type=str, default=None)
    parser.add_argument("--lyrics", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_k_layers", type=str, default=None, help="Comma-separated 8 ints, e.g. 47,47,47,45,39,37,38,39")
    parser.add_argument("--k_macro", nargs=3, type=int, default=None, metavar=("FUNDAMENTAL", "ACOUSTIC", "FINE"))
    parser.add_argument("--ar_cfg", dest="ar_guidance_scale", type=float, default=None, help="Stage 1 AR CFG scale.")
    parser.add_argument("--inst_scheduler", dest="instrumental_scheduler", type=str, choices=SUPPORTED_SOLVERS, default=None)
    parser.add_argument("--voc_scheduler", dest="vocal_scheduler", type=str, choices=SUPPORTED_SOLVERS, default=None)
    parser.add_argument("--steps", dest="num_inference_steps", type=int, default=None)
    parser.add_argument("--inst_cfg", dest="instrumental_guidance_scale", type=float, default=None, help="Instrumental CFG scale.")
    parser.add_argument("--voc_cfg", dest="vocal_guidance_scale", type=float, default=None, help="Vocal CFG scale.")
    parser.add_argument("--eta", dest="eta", type=float, default=None, help="SDE stochasticity scale (0.0 to 1.0).")
    parser.add_argument("--s_noise", dest="s_noise", type=float, default=None, help="SDE noise multiplier.")
    parser.add_argument("--noise_topology", type=str, choices=SUPPORTED_NOISE_TOPOLOGIES, default=None)
    parser.add_argument("--blue_noise_alpha", type=float, default=None)
    parser.add_argument("--enable_pm_diffusion", action="store_true", default=None)
    parser.add_argument("--pm_iterations", type=int, default=None)
    parser.add_argument("--pm_conductance", type=float, default=None)
    parser.add_argument("--pm_lambda", type=float, default=None)
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
    if args.instrumental_scheduler is not None:
        req.instrumental_scheduler = args.instrumental_scheduler
    if args.vocal_scheduler is not None:
        req.vocal_scheduler = args.vocal_scheduler
    if args.num_inference_steps is not None:
        req.num_inference_steps = args.num_inference_steps
    if args.instrumental_guidance_scale is not None:
        req.instrumental_guidance_scale = args.instrumental_guidance_scale
    if args.vocal_guidance_scale is not None:
        req.vocal_guidance_scale = args.vocal_guidance_scale
    if args.eta is not None:
        req.eta = args.eta
    if args.s_noise is not None:
        req.s_noise = args.s_noise
    if args.noise_topology is not None:
        req.noise_topology = args.noise_topology
    if args.blue_noise_alpha is not None:
        req.blue_noise_alpha = args.blue_noise_alpha
    if args.enable_pm_diffusion is not None:
        req.enable_pm_diffusion = args.enable_pm_diffusion
    if args.pm_iterations is not None:
        req.pm_iterations = args.pm_iterations
    if args.pm_conductance is not None:
        req.pm_conductance = args.pm_conductance
    if args.pm_lambda is not None:
        req.pm_lambda = args.pm_lambda
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
    if args.save_preset:
        target = Path(args.save_preset)
        if not target.is_absolute() and target.name == DEFAULT_PRESET_FILENAME:
            target = ROOT_DIR / DEFAULT_PRESET_FILENAME
        req.save_preset(target)
        print(f"Preset exported to {target}")
        sys.exit(0)

    if not args.batch:
        run_interactive_harness(engine=None, req=req)
    else:
        engine = MusicEngine(repo_id=req.repo_id, device=req.device)
        resp = engine.synthesize(req)
        print_telemetry(resp)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)