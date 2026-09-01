import os
import sys
import warnings
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import argparse
import traceback
from typing import Optional
from schema import (
    GenerationRequest,
    GenerationResponse,
    SUPPORTED_SCHEDULERS,
    SUPPORTED_NOISE_TOPOLOGIES,
)
from engine import MusicEngine


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
    print(f"ODE Solver Trajectory: {resp.scheduler_used.upper()}")
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
    t_disp = f"{req.temperature:.4f}" if req.temperature is not None else "0.9100"
    p_disp = f"{req.top_p:.4f}" if req.top_p is not None else "0.9600"
    k_disp = f"{req.top_k}" if req.top_k is not None else "44"
    ar_cfg_disp = f"{req.ar_guidance_scale:.4f}" if req.ar_guidance_scale is not None else "1.5200"
    steps_disp = f"{req.num_inference_steps}" if req.num_inference_steps is not None else "42"
    dit_cfg_disp = f"{req.guidance_scale:.4f}" if req.guidance_scale is not None else "1.7800"
    declick_disp = "ENABLED (Symmetric Hann)" if req.apply_declick else "DISABLED"
    offload_disp = "ENABLED (Sequential Streaming)" if req.cpu_offload else "DISABLED (Resident VRAM)"
    pm_disp = (
        f"ENABLED (Iters={req.pm_iterations}, K={req.pm_conductance:.4f}, Lambda={req.pm_lambda:.4f})"
        if req.enable_pm_diffusion
        else "DISABLED"
    )

    print("\n" + "=" * 84)
    print("               MINIMAX-MUSIC3 MODALITY EXPLORATION & ABLATION HARNESS")
    print("=" * 84)
    print(" --- AESTHETIC & SEMANTIC CONDITIONING ---")
    print(f" [1]  Genre:                 {req.genre}")
    print(f" [2]  BPM:                   {req.bpm}")
    print(f" [3]  Key Signature:         {req.key}")
    print(f" [4]  Mood Narrative:        {req.mood}")
    print(f" [5]  Vocal Architecture:    {req.vocals}")
    print(f" [6]  Arrangement Details:   {req.arrangement}")
    print(f" [7]  Raw Prompt Override:   {req.raw_prompt if req.raw_prompt else '<Auto-Compiled Metadata>'}")
    print(" --- STAGE 1 AUTOREGRESSIVE GENERATION ---")
    print(f" [8]  Temperature & AR CFG:  T: {t_disp} | AR CFG: {ar_cfg_disp}")
    print(f" [9]  Nucleus Top-P / Top-K: Top-P: {p_disp} | Top-K: {k_disp}")
    print(" --- STAGE 2 CONTINUOUS FLOW-MATCHING & PDE REGULARIZATION ---")
    print(f" [10] ODE Solver Trajectory: {req.scheduler_type.upper()}")
    print(f" [11] Inference Steps / DiT: Steps: {steps_disp} | DiT Guidance: {dit_cfg_disp}")
    print(
        f" [12] Latent Prior Topology: {req.noise_topology.upper()}{f' (Alpha: {req.blue_noise_alpha:.4f})' if req.noise_topology == 'blue_noise' else ''}"
    )
    print(f" [13] 1D Temporal PM PDE:    {pm_disp}")
    print(" --- TEMPORAL SYNTHESIS & HARDWARE MEMORY ---")
    print(f" [14] Track Length Ceiling:  {req.audio_duration:.4f}s")
    print(f" [15] PRNG Generation Seed:  {req.seed}")
    print(f" [16] Output WAV Destination:{req.output_path}")
    print(f" [17] Edit Structured Lyrics ({len(req.lyrics.splitlines())} lines configured)")
    print(f" [18] DSP Boundary De-Click: {declick_disp}")
    print(f" [19] Memory CPU Streaming:  {offload_disp}")
    print("-" * 84)
    print(" [P] Print Prompt   [L] Load Preset (JSON)   [S] Save Preset (JSON)")
    print(" [G] Generate Audio [Q] Quit")
    print("=" * 84)


def edit_multiline_lyrics(current_lyrics: str) -> str:
    print("\n--- Edit Structured Lyrics Markup ---")
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


def run_interactive_harness(engine: Optional[MusicEngine], req: GenerationRequest) -> None:
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
            b = input(f"Enter BPM (30 - 300) [{req.bpm}]: ").strip()
            if b.isdigit() and 30 <= int(b) <= 300:
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
                f"Enter Sampling Temperature [{req.temperature if req.temperature is not None else 0.9100}]: "
            ).strip()
            req.temperature = float(t) if t and t.lower() != "native" else None
            ar_g = input(
                f"Enter Stage 1 AR Guidance Scale (CFG) [{req.ar_guidance_scale if req.ar_guidance_scale is not None else 1.5200}]: "
            ).strip()
            req.ar_guidance_scale = float(ar_g) if ar_g and ar_g.lower() != "native" else None
        elif choice == "9":
            p = input(f"Enter Top-P [{req.top_p if req.top_p is not None else 0.9600}]: ").strip()
            req.top_p = float(p) if p and p.lower() != "native" else None
            k = input(f"Enter Top-K [{req.top_k if req.top_k is not None else 44}]: ").strip()
            req.top_k = int(k) if k and k.lower() != "native" else None
        elif choice == "10":
            print("\n[1] HEUN (2nd-Order Predictor-Corrector)  [2] EULER (1st-Order Forward)  [3] NATIVE")
            s_map = {"1": "heun", "heun": "heun", "2": "euler", "euler": "euler", "3": "native", "native": "native"}
            sel = input(f"Select solver [{req.scheduler_type}]: ").strip().lower()
            req.scheduler_type = s_map.get(sel, req.scheduler_type)
        elif choice == "11":
            s = input(
                f"Enter Steps [{req.num_inference_steps if req.num_inference_steps is not None else 42}]: "
            ).strip()
            req.num_inference_steps = int(s) if s and s.lower() != "native" else None
            c = input(
                f"Enter Stage 2 DiT Guidance Scale [{req.guidance_scale if req.guidance_scale is not None else 1.7800}]: "
            ).strip()
            req.guidance_scale = float(c) if c and c.lower() != "native" else None
        elif choice == "12":
            print("\n[1] BLUE_NOISE (High-Pass |f|^alpha)  [2] GAUSSIAN (Standard Normal)")
            n_sel = input(f"Select Noise Topology [{req.noise_topology}]: ").strip()
            if n_sel in ["1", "blue_noise"]:
                req.noise_topology = "blue_noise"
                a_val = input(f"Enter Blue Noise Alpha [0.0 - 2.0] [{req.blue_noise_alpha:.4f}]: ").strip()
                if a_val:
                    req.blue_noise_alpha = float(a_val)
            elif n_sel in ["2", "gaussian"]:
                req.noise_topology = "gaussian"
        elif choice == "13":
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
        elif choice == "14":
            d = input(f"Enter Duration Ceiling (s) [{req.audio_duration:.4f}]: ").strip()
            if d:
                req.audio_duration = float(d)
        elif choice == "15":
            sd = input(f"Enter PRNG Seed [{req.seed}]: ").strip()
            if sd:
                req.seed = int(sd)
        elif choice == "16":
            o = input(f"Enter Output WAV Path [{req.output_path}]: ").strip()
            if o:
                req.output_path = o
        elif choice == "17":
            req.lyrics = edit_multiline_lyrics(req.lyrics)
        elif choice == "18":
            req.apply_declick = not req.apply_declick
        elif choice == "19":
            req.cpu_offload = not req.cpu_offload
        elif choice == "P":
            print(f"\n--- Compiled Prompt ---\n{req.compile_prompt()}\n")
            input("Press Enter to continue...")
        elif choice == "L":
            p_path = input("Enter JSON preset to load: ").strip()
            try:
                req = GenerationRequest.load_preset(Path(p_path))
                print(f"Preset loaded successfully from {p_path}")
            except Exception as e:
                print(f"Preset load error: {e}")
        elif choice == "S":
            p_path = input("Enter destination JSON preset path: ").strip()
            try:
                req.save_preset(Path(p_path))
                print(f"Preset saved to {p_path}")
            except Exception as e:
                print(f"Preset save error: {e}")
        elif choice == "G":
            if engine is None:
                print("\nInitializing neural engine...")
                engine = MusicEngine(repo_id=req.repo_id, device=req.device)
            resolved_ar_cfg = req.ar_guidance_scale if req.ar_guidance_scale is not None else 1.5200
            resolved_top_k = req.top_k if req.top_k is not None else 44
            resolved_dit_cfg = req.guidance_scale if req.guidance_scale is not None else 1.7800
            print(
                f"\nExecuting Synthesis Pass (Ceiling={req.audio_duration:.4f}s, AR_CFG={resolved_ar_cfg:.4f}, TopK={resolved_top_k}, DiT_CFG={resolved_dit_cfg:.4f}, Solver={req.scheduler_type.upper()}, Noise={req.noise_topology}, PM={req.enable_pm_diffusion})..."
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
    parser.add_argument("--ar_cfg", dest="ar_guidance_scale", type=float, default=None, help="Stage 1 AR CFG scale.")
    parser.add_argument("--scheduler", dest="scheduler_type", type=str, choices=SUPPORTED_SCHEDULERS, default=None)
    parser.add_argument("--steps", dest="num_inference_steps", type=int, default=None)
    parser.add_argument("--cfg", "--dit_cfg", dest="guidance_scale", type=float, default=None, help="Stage 2 DiT CFG scale.")
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

    req = GenerationRequest.load_preset(Path(args.load_preset)) if args.load_preset else GenerationRequest()

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
    if args.ar_guidance_scale is not None:
        req.ar_guidance_scale = args.ar_guidance_scale
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
    except Exception:
        traceback.print_exc()
        sys.exit(1)