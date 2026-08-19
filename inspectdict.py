#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / ".hf_cache"

import torch
from safetensors.torch import load_file as load_safetensors

def get_dtype_bytes(dtype_str: str) -> float:
    dtype_map = {
        "float32": 4.0, "fp32": 4.0,
        "float16": 2.0, "fp16": 2.0,
        "bfloat16": 2.0, "bf16": 2.0,
        "float8_e4m3fn": 1.0, "e4m3fn": 1.0,
        "float8_e5m2": 1.0, "e5m2": 1.0,
        "float8_e8m0fnu": 1.0,
        "int8": 1.0, "uint8": 1.0,
        "int64": 8.0, "long": 8.0,
        "int32": 4.0, "bool": 0.125
    }
    cleaned = str(dtype_str).replace("torch.", "").lower()
    return dtype_map.get(cleaned, 2.0)

def locate_snapshot_root(cache_root: Path) -> Path:
    snapshots = list(cache_root.glob("**/snapshots/*"))
    if not snapshots:
        raise FileNotFoundError(f"No snapshot directories located inside cache anchor: {cache_root}")
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshots[0]

def inspect_safetensors_file(file_path: Path) -> List[Dict[str, Any]]:
    records = []
    state_dict = load_safetensors(str(file_path), device="cpu")
    for k, v in state_dict.items():
        numel = v.numel()
        dtype_str = str(v.dtype).replace("torch.", "")
        bpe = get_dtype_bytes(dtype_str)
        size_mb = (numel * bpe) / (1024 ** 2)
        records.append({
            "key": k,
            "shape": list(v.shape),
            "numel": numel,
            "dtype": dtype_str,
            "size_mb": size_mb,
            "is_floating": v.is_floating_point(),
            "file": file_path.name,
            "format": "safetensors"
        })
    return records

def inspect_bin_file(file_path: Path) -> List[Dict[str, Any]]:
    records = []
    try:
        state_dict = torch.load(str(file_path), map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict):
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    numel = v.numel()
                    dtype_str = str(v.dtype).replace("torch.", "")
                    bpe = get_dtype_bytes(dtype_str)
                    size_mb = (numel * bpe) / (1024 ** 2)
                    records.append({
                        "key": k,
                        "shape": list(v.shape),
                        "numel": numel,
                        "dtype": dtype_str,
                        "size_mb": size_mb,
                        "is_floating": v.is_floating_point(),
                        "file": file_path.name,
                        "format": "pytorch_bin"
                    })
    except Exception as e:
        records.append({
            "key": f"<ERROR: {file_path.name}>",
            "shape": [],
            "numel": 0,
            "dtype": "unknown",
            "size_mb": file_path.stat().st_size / (1024 ** 2),
            "is_floating": False,
            "file": file_path.name,
            "format": "unreadable"
        })
    return records

def profile_snapshot(snapshot_dir: Path, module_filter: str = "") -> None:
    print("\n" + "=" * 90)
    print(f"SNAPSHOT INSPECTOR: {snapshot_dir.name}")
    print(f"PATH: {snapshot_dir}")
    print("=" * 90)

    safetensors_paths = list(snapshot_dir.rglob("*.safetensors"))
    bin_paths = list(snapshot_dir.rglob("*.bin")) + list(snapshot_dir.rglob("*.pt"))

    module_records = defaultdict(list)
    duplicate_bloat_mb = 0.0

    # Detect duplicate formats
    sf_stems = {p.stem for p in safetensors_paths}
    for bp in bin_paths:
        if bp.stem in sf_stems or bp.stem.replace("pytorch_model", "model") in sf_stems:
            duplicate_bloat_mb += bp.stat().st_size / (1024 ** 2)

    all_files = safetensors_paths + bin_paths
    for fp in all_files:
        rel_parent = fp.parent.relative_to(snapshot_dir)
        module_name = str(rel_parent) if str(rel_parent) != "." else "root"

        if module_filter and module_filter.lower() not in module_name.lower():
            continue

        if fp.suffix == ".safetensors":
            recs = inspect_safetensors_file(fp)
        else:
            recs = inspect_bin_file(fp)

        module_records[module_name].extend(recs)

    total_params = 0
    total_size_mb = 0.0
    dtype_breakdown = defaultdict(lambda: {"count": 0, "params": 0, "size_mb": 0.0})

    print(f"\n{'MODULE NAMESPACE':<35} | {'PARAMS':<12} | {'DISK SIZE':<10} | {'PRIMARY DTYPES'}")
    print("-" * 90)

    for mod_name, recs in sorted(module_records.items()):
        mod_params = sum(r["numel"] for r in recs)
        mod_size = sum(r["size_mb"] for r in recs)
        dtypes = sorted(list({r["dtype"] for r in recs}))
        dtype_summary = ", ".join(dtypes)

        total_params += mod_params
        total_size_mb += mod_size

        for r in recs:
            dtype_breakdown[r["dtype"]]["count"] += 1
            dtype_breakdown[r["dtype"]]["params"] += r["numel"]
            dtype_breakdown[r["dtype"]]["size_mb"] += r["size_mb"]

        params_str = f"{mod_params / 1e6:,.2f} M" if mod_params < 1e9 else f"{mod_params / 1e9:,.2f} B"
        print(f"{mod_name:<35} | {params_str:<12} | {mod_size:>8.2f} MB | {dtype_summary}")

    print("-" * 90)
    total_params_str = f"{total_params / 1e9:,.3f} B" if total_params >= 1e9 else f"{total_params / 1e6:,.2f} M"
    print(f"{'TOTAL ACTIVE PARAMETERS':<35} | {total_params_str:<12} | {total_size_mb / 1024:>7.2f} GB |")
    print("=" * 90)

    print("\nPRECISION ARCHITECTURE BREAKDOWN:")
    print(f"{'DATA TYPE':<20} | {'TENSORS':<10} | {'PARAMS':<15} | {'MEMORY / DISK'}")
    print("-" * 65)
    for dt, stats in sorted(dtype_breakdown.items(), key=lambda x: x[1]["size_mb"], reverse=True):
        p_str = f"{stats['params'] / 1e6:,.2f} M" if stats['params'] < 1e9 else f"{stats['params'] / 1e9:,.2f} B"
        print(f"{dt:<20} | {stats['count']:<10} | {p_str:<15} | {stats['size_mb']:>9.2f} MB")
    print("-" * 65)

    if duplicate_bloat_mb > 0:
        print(f"\n[!] REDUNDANCY DETECTED: {duplicate_bloat_mb:>8.2f} MB of duplicate .bin/.pt files mirror .safetensors shards.")
        print("    These can be safely purged without affecting inference.")

def dump_verbose_table(snapshot_dir: Path, module_filter: str = "") -> None:
    safetensors_paths = list(snapshot_dir.rglob("*.safetensors"))
    for fp in safetensors_paths:
        rel_parent = fp.parent.relative_to(snapshot_dir)
        module_name = str(rel_parent) if str(rel_parent) != "." else "root"
        if module_filter and module_filter.lower() not in module_name.lower():
            continue

        print(f"\n>>> Container: {fp.relative_to(snapshot_dir)}")
        print(f"{'PARAMETER KEY':<55} | {'SHAPE':<22} | {'DTYPE':<14} | {'SIZE'}")
        print("-" * 105)
        recs = inspect_safetensors_file(fp)
        for r in recs:
            shape_str = str(r["shape"])
            print(f"{r['key']:<55} | {shape_str:<22} | {r['dtype']:<14} | {r['size_mb']:>7.3f} MB")

def main():
    parser = argparse.ArgumentParser(description="Profile tensor keys, dimensions, dtypes, and bloat in model snapshots.")
    parser.add_argument("--cache_dir", type=str, default=str(CACHE_DIR), help="Root directory of the Hugging Face cache.")
    parser.add_argument("--module", type=str, default="", help="Filter by specific module namespace (e.g. vocoder, language_model).")
    parser.add_argument("--verbose", action="store_true", help="Print granular layer-by-layer tensor mapping.")
    args = parser.parse_args()

    cache_root = Path(args.cache_dir)
    if not cache_root.exists():
        print(f"Error: Cache directory not found at {cache_root}")
        sys.exit(1)

    try:
        snapshot_dir = locate_snapshot_root(cache_root)
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    if args.verbose:
        dump_verbose_table(snapshot_dir, args.module)
    else:
        profile_snapshot(snapshot_dir, args.module)

if __name__ == "__main__":
    main()