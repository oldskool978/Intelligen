#!/usr/bin/env python3
import os
import sys
import warnings
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*There are modules in.*should be kept in float32.*")
warnings.filterwarnings("ignore", message=".*Modular Diffusers is currently an experimental feature.*")
warnings.filterwarnings("ignore", message=".*Guiders are currently an experimental feature.*")

import argparse
import traceback
from typing import Optional

from schema import (
    GenerationRequest,
    GenerationResponse,
    SUPPORTED_SCHEDULERS,
    SUPPORTED_NOISE_TOPOLOGIES
)
from engine import MusicEngine

def print_telemetry(resp: GenerationResponse) -> None:
    print("\n" + "=" * 80)
    print("                      GENERATION TELEMETRY REPORT")
    print("=" * 80)
    print(f"Master Output:        {resp.output_path}")
    print(f"Sampling Rate:        {resp.sample_rate} Hz (32-bit Float PCM)")
    print(f"Audio Length:         {resp.duration_seconds:.2f}s ({resp.total_samples:,} samples)")
    print(f"Compute Time:         {resp.generation_time_seconds:.2f}s (RTF: {resp.real_time_factor:.3f}x)")
    print(f"ODE Solver:           {resp.scheduler_used.upper()}")
    print(f"Noise Topology:       {resp.noise_topology_used}")
    print(f"Peak Level:           {resp.peak_linear:.6f} ({resp.peak_dbfs:.2f} dBFS)")
    print(f"Integrated RMS:       {resp.rms_dbfs:.2f} dBFS")
    print("-" * 80)
    print(f"Effective Conditioning Prompt:\n{resp.effective_prompt}")
    print("=" * 80 + "\n")

def display_menu(req: GenerationRequest) -> None:
    t_disp = f"{req.temperature:.2f}" if req.temperature is not None else "<Native Default>"
    p_disp = f"{req.top_p:.2f}" if req.top_p is not None else "<Native Default>"
    k_disp = f"{req.top_k}" if req.top_k is not None else "<Native Default>"
    steps_disp = f"{req.num_inference_steps}" if req.num_inference_steps is not None else "<Native Default>"
    cfg_disp = f"{req.guidance_scale:.2f}" if req.guidance_scale is not None else "<Native Default>"

    print("\n" + "=" * 78)
    print("                 MINIMAX-MUSIC3 TEST & REFINEMENT HARNESS")
    print("=" * 78)
    print(f" [1]  Genre:                 {req.genre}")
    print(f" [2]  BPM:                   {req.bpm}")
    print(f" [3]  Key:                   {req.key}")
    print(f" [4]  Mood:                  {req.mood}")
    print(f" [5]  Vocals:                {req.vocals}")
    print(f" [6]  Arrangement:           {req.arrangement}")
    print(f" [7]  Raw Prompt:            {req.raw_prompt if req.raw_prompt else '<Auto-Compiled>'}")
    print("-" * 78)
    print(f" [8]  Temperature:           {t_disp}")
    print(f" [9]  Top-P:                 {p_disp}")
    print(f" [10] Top-K:                 {k_disp}")
    print("-" * 78)
    print(f" [11] ODE Scheduler:         {req.scheduler_type.upper()}")
    print(f" [12] Inference Steps:       {steps_disp}")
    print(f" [13] Guidance Scale (CFG):  {cfg_disp}")
    print("-" * 78)
    print(f" [14] Noise Topology:        {req.noise_topology}")
    if req.noise_topology == "blue_noise":
        print(f" [15] Blue Noise Alpha:      {req.blue_noise_alpha:.2f}")
    elif req.noise_topology == "perona_malik":
        print(f" [15] PM Parameters:         Iters={req.pm_iterations}, K={req.pm_conductance:.2f}, Lambda={req.pm_lambda:.2f}")
    else:
        print(f" [15] Topology Config:       <Standard Gaussian N(0, I)>")
    print("-" * 78)
    print(f" [16] Duration:              {req.audio_duration}s")
    print(f" [17] Seed:                  {req.seed}")
    print(f" [18] Output File:           {req.output_path}")
    print(f" [19] Edit Lyrics            ({len(req.lyrics.splitlines())} lines configured)")
    print("-" * 78)
    print(" [P] Print Compiled Prompt   [L] Load Preset (JSON)   [S] Save Preset (JSON)")
    print(" [G] Generate Audio          [Q] Quit")
    print("=" * 78)

def edit_multiline_lyrics(current_lyrics: str) -> str:
    print("\n--- Edit Lyrics Markup ---")
    print("Current Lyrics:")
    print(current_lyrics)
    print("\nEnter new lyrics (Type '__DONE__' on an empty line to finish):")
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "__DONE__":
                break
            lines.append(line)
        except EOFError:
            break
    new_text = "\n".join(lines).strip()
    return new_text if new_text else current_lyrics

def prompt_temperature(current: Optional[float]) -> Optional[float]:
    print("\n--- Configure Sampling Temperature (Stage 1 LM Logits) ---")
    print(" Enter 'native' to restore native pipeline default.")
    val = input(f"Enter Temperature [{current if current is not None else 'native'}]: ").strip()
    if not val or val.lower() == "native":
        return None
    try:
        t = float(val)
        if 0.01 <= t <= 3.0:
            return t
    except ValueError:
        pass
    return current

def prompt_top_p(current: Optional[float]) -> Optional[float]:
    print("\n--- Configure Nucleus Sampling Top-P (Stage 1 LM Filtering) ---")
    print(" Enter 'native' to restore native pipeline default.")
    val = input(f"Enter Top-P [{current if current is not None else 'native'}]: ").strip()
    if not val or val.lower() == "native":
        return None
    try:
        p = float(val)
        if 0.01 <= p <= 1.0:
            return p
    except ValueError:
        pass
    return current

def prompt_top_k(current: Optional[int]) -> Optional[int]:
    print("\n--- Configure Top-K Candidate Pool (Stage 1 LM Truncation) ---")
    print(" Enter 'native' to restore native pipeline default.")
    val = input(f"Enter Top-K [{current if current is not None else 'native'}]: ").strip()
    if not val or val.lower() == "native":
        return None
    try:
        k = int(val)
        if 1 <= k <= 500:
            return k
    except ValueError:
        pass
    return current

def prompt_scheduler(current: str) -> str:
    print("\n--- Select Flow-Matching ODE Solver ---")
    print(" [1] Native (Model default scheduler)")
    print(" [2] Euler  (1st-Order Forward Euler)")
    print(" [3] Heun   (2nd-Order Predictor-Corrector)")
    val = input(f"Select choice [1-3] (Current: {current}): ").strip().lower()
    mapping = {
        "1": "native", "native": "native",
        "2": "euler", "euler": "euler",
        "3": "heun", "heun": "heun"
    }
    return mapping.get(val, current)

def prompt_steps(current: Optional[int]) -> Optional[int]:
    print("\n--- Configure ODE Inference Steps ---")
    print(" Enter 'native' to restore native pipeline default.")
    val = input(f"Enter Inference Steps [{current if current is not None else 'native'}]: ").strip()
    if not val or val.lower() == "native":
        return None
    try:
        s = int(val)
        if 1 <= s <= 200:
            return s
    except ValueError:
        pass
    return current

def prompt_cfg(current: Optional[float]) -> Optional[float]:
    print("\n--- Configure Classifier-Free Guidance (CFG Scale) ---")
    print(" Enter 'native' to restore native pipeline default.")
    val = input(f"Enter Guidance Scale [{current if current is not None else 'native'}]: ").strip()
    if not val or val.lower() == "native":
        return None
    try:
        c = float(val)
        if 0.0 <= c <= 20.0:
            return c
    except ValueError:
        pass
    return current

def prompt_noise_topology(current: str) -> str:
    print("\n--- Select Initial Latent Manifold Prior (x1 Boundary) ---")
    print(" [1] Gaussian      - Native flat-spectrum standard normal N(0, I)")
    print(" [2] Blue Noise    - High-pass spectral tilt (|f|^alpha)")
    print(" [3] Perona-Malik  - Anisotropic PDE diffusion")
    val = input(f"Select choice [1-3] (Current: {current}): ").strip().lower()
    mapping = {
        "1": "gaussian", "gaussian": "gaussian",
        "2": "blue_noise", "blue_noise": "blue_noise",
        "3": "perona_malik", "perona_malik": "perona_malik"
    }
    return mapping.get(val, current)

def prompt_topology_parameters(req: GenerationRequest) -> None:
    if req.noise_topology == "blue_noise":
        print("\n--- Configure Blue Noise Spectral Exponent (Alpha) ---")
        val = input(f"Enter Alpha [0.0 - 2.0] [{req.blue_noise_alpha}]: ").strip()
        try:
            a = float(val)
            if 0.0 <= a <= 2.0:
                req.blue_noise_alpha = a
        except ValueError:
            pass
    elif req.noise_topology == "perona_malik":
        print("\n--- Configure Perona-Malik Anisotropic PDE Parameters ---")
        val_i = input(f" [1] Iterations [1 - 30] [{req.pm_iterations}]: ").strip()
        val_k = input(f" [2] Conductance K [0.01 - 5.0] [{req.pm_conductance}]: ").strip()
        val_l = input(f" [3] Lambda factor [0.01 - 0.25] [{req.pm_lambda}]: ").strip()
        try:
            if val_i and 1 <= int(val_i) <= 30:
                req.pm_iterations = int(val_i)
            if val_k and 0.01 <= float(val_k) <= 5.0:
                req.pm_conductance = float(val_k)
            if val_l and 0.01 <= float(val_l) <= 0.25:
                req.pm_lambda = float(val_l)
        except ValueError:
            pass
    else:
        print("\nStandard Gaussian topology requires no parameter adjustments.")
        input("Press Enter to continue...")

def run_interactive_harness(engine: Optional[MusicEngine], req: GenerationRequest) -> None:
    while True:
        display_menu(req)
        choice = input("Select an option: ").strip().upper()

        if choice == "1":
            val = input(f"Enter Genre [{req.genre}]: ").strip()
            if val:
                req.genre = val
        elif choice == "2":
            val = input(f"Enter BPM (30 - 300) [{req.bpm}]: ").strip()
            if val.isdigit() and 30 <= int(val) <= 300:
                req.bpm = int(val)
        elif choice == "3":
            val = input(f"Enter Musical Key [{req.key}]: ").strip()
            if val:
                req.key = val
        elif choice == "4":
            val = input(f"Enter Mood/Style Narrative [{req.mood}]: ").strip()
            if val:
                req.mood = val
        elif choice == "5":
            val = input(f"Enter Vocal Profile [{req.vocals}]: ").strip()
            if val:
                req.vocals = val
        elif choice == "6":
            val = input(f"Enter Arrangement Details [{req.arrangement}]: ").strip()
            if val:
                req.arrangement = val
        elif choice == "7":
            val = input("Enter Raw Prompt override (leave empty to reset to Auto-Compiled): ").strip()
            req.raw_prompt = val if val else None
        elif choice == "8":
            req.temperature = prompt_temperature(req.temperature)
        elif choice == "9":
            req.top_p = prompt_top_p(req.top_p)
        elif choice == "10":
            req.top_k = prompt_top_k(req.top_k)
        elif choice == "11":
            req.scheduler_type = prompt_scheduler(req.scheduler_type)
        elif choice == "12":
            req.num_inference_steps = prompt_steps(req.num_inference_steps)
        elif choice == "13":
            req.guidance_scale = prompt_cfg(req.guidance_scale)
        elif choice == "14":
            req.noise_topology = prompt_noise_topology(req.noise_topology)
        elif choice == "15":
            prompt_topology_parameters(req)
        elif choice == "16":
            val = input(f"Enter Duration in seconds (0.1 - 47.5) [{req.audio_duration}]: ").strip()
            try:
                dur = float(val)
                if 0.0 < dur <= 47.5:
                    req.audio_duration = dur
                else:
                    print("Duration must be between 0.1 and 47.5 seconds.")
            except ValueError:
                pass
        elif choice == "17":
            val = input(f"Enter PRNG Seed (-1 for random) [{req.seed}]: ").strip()
            try:
                req.seed = int(val)
            except ValueError:
                pass
        elif choice == "18":
            val = input(f"Enter Output Audio Filename [{req.output_path}]: ").strip()
            if val:
                req.output_path = val
        elif choice == "19":
            req.lyrics = edit_multiline_lyrics(req.lyrics)
        elif choice == "P":
            print("\n--- Compiled Conditioning Prompt ---")
            print(req.compile_prompt())
            input("\nPress Enter to continue...")
        elif choice == "L":
            preset_path = input("Enter JSON preset path to load: ").strip()
            try:
                req = GenerationRequest.load_preset(Path(preset_path))
                print(f"Loaded preset successfully from {preset_path}")
            except Exception as e:
                print(f"Failed to load preset: {e}")
        elif choice == "S":
            preset_path = input("Enter destination JSON preset path: ").strip()
            try:
                req.save_preset(Path(preset_path))
                print(f"Saved preset successfully to {preset_path}")
            except Exception as e:
                print(f"Failed to save preset: {e}")
        elif choice == "G":
            if engine is None:
                print("\nInitializing neural engine in VRAM...")
                engine = MusicEngine(repo_id=req.repo_id, device=req.device)
            print(f"\nSynthesizing track ({req.audio_duration}s, seed={req.seed}, solver={req.scheduler_type.upper()}, noise={req.noise_topology})...")
            try:
                resp = engine.synthesize(req)
                print_telemetry(resp)
            except Exception as e:
                print(f"Synthesis failed: {e}", file=sys.stderr)
        elif choice == "Q":
            print("Exiting harness.")
            sys.exit(0)

def main() -> None:
    parser = argparse.ArgumentParser(description="Modular CLI and Test Harness for MiniMax-Music3.")
    parser.add_argument("--batch", action="store_true", help="Run in headless non-interactive batch mode.")
    parser.add_argument("--genre", type=str, default=None, help="Target musical genre.")
    parser.add_argument("--bpm", type=int, default=None, help="Tempo in beats per minute.")
    parser.add_argument("--key", type=str, default=None, help="Musical root key and mode.")
    parser.add_argument("--mood", type=str, default=None, help="Atmosphere and dynamic progression.")
    parser.add_argument("--vocals", type=str, default=None, help="Vocal timbre and acoustic traits.")
    parser.add_argument("--arrangement", type=str, default=None, help="Instrumentation and mix elements.")
    parser.add_argument("--raw_prompt", type=str, default=None, help="Direct prompt override.")
    parser.add_argument("--lyrics", type=str, default=None, help="Lyrical markup text or path to .txt file.")
    
    parser.add_argument("--temperature", type=float, default=None, help="Autoregressive LM sampling temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Autoregressive LM nucleus sampling top-p.")
    parser.add_argument("--top_k", type=int, default=None, help="Autoregressive LM top-k candidate truncation.")
    
    parser.add_argument("--scheduler", dest="scheduler_type", type=str, choices=SUPPORTED_SCHEDULERS, default=None, help="Flow-matching ODE solver.")
    parser.add_argument("--steps", "--num_inference_steps", dest="num_inference_steps", type=int, default=None, help="ODE trajectory inference steps.")
    parser.add_argument("--cfg", "--guidance_scale", dest="guidance_scale", type=float, default=None, help="Classifier-free guidance scale.")
    
    parser.add_argument("--noise_topology", type=str, choices=SUPPORTED_NOISE_TOPOLOGIES, default=None, help="Initial latent manifold topology.")
    parser.add_argument("--blue_noise_alpha", type=float, default=None, help="Blue noise spectral tilt exponent.")
    parser.add_argument("--pm_iterations", type=int, default=None, help="Perona-Malik diffusion iterations.")
    parser.add_argument("--pm_conductance", type=float, default=None, help="Perona-Malik conductance threshold K.")
    parser.add_argument("--pm_lambda", type=float, default=None, help="Perona-Malik stability factor lambda.")
    
    parser.add_argument("--duration", type=float, default=None, help="Audio length in seconds.")
    parser.add_argument("--seed", type=int, default=None, help="PRNG seed.")
    parser.add_argument("--output", type=str, default=None, help="Output destination WAV path.")
    parser.add_argument("--load_preset", type=str, default=None, help="Load parameters from a JSON preset.")
    parser.add_argument("--save_preset", type=str, default=None, help="Export parameters to a JSON preset and exit.")
    parser.add_argument("--device", type=str, default="cuda", help="Execution provider (cuda/cpu).")
    parser.add_argument("--repo_id", type=str, default="MiniMaxAI/MiniMax-Music3", help="Hugging Face repo identifier.")

    args = parser.parse_args()

    if args.load_preset:
        req = GenerationRequest.load_preset(Path(args.load_preset))
    else:
        req = GenerationRequest()

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
    if args.scheduler_type is not None:
        req.scheduler_type = args.scheduler_type
    if args.num_inference_steps is not None:
        req.num_inference_steps = args.num_inference_steps
    if args.guidance_scale is not None:
        req.guidance_scale = args.guidance_scale
    if args.noise_topology is not None:
        req.noise_topology = args.noise_topology
    if args.blue_noise_alpha is not None:
        req.blue_noise_alpha = args.blue_noise_alpha
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
    if args.device is not None:
        req.device = args.device
    if args.repo_id is not None:
        req.repo_id = args.repo_id

    if args.lyrics is not None:
        lyrics_path = Path(args.lyrics)
        if lyrics_path.is_file():
            req.lyrics = lyrics_path.read_text(encoding="utf-8")
        else:
            req.lyrics = args.lyrics

    if args.save_preset:
        req.save_preset(Path(args.save_preset))
        print(f"Preset exported to {args.save_preset}")
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
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)