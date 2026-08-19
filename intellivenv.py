#!/usr/bin/env python3
import os
import sys
import re
import shutil
import zipfile
import tarfile
import urllib.request
import subprocess
import stat
from pathlib import Path

PREDEFINED_MATRIX = {
    "1": "3.10.11",
    "2": "3.11.9",
    "3": "3.12.3",
    "4": "3.13.13",
}

STANDALONE_RELEASE = "20260623"

AMD_GFX_RANK = {
    "gfx950": 100,
    "gfx942": 95,
    "gfx90a": 90,
    "gfx120X": 85,
    "gfx1100": 80,
    "gfx1101": 75,
    "gfx1102": 70,
    "gfx1030": 65,
    "gfx1103": 50,
    "gfx1035": 45,
}


def print_status(msg: str, status: str = "INFO") -> None:
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARN": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m",
    }
    if os.name == "nt" and not os.environ.get("WT_SESSION"):
        print(f"[{status}] {msg}")
    else:
        print(f"{colors.get(status, '')}[{status}] {msg}{colors['RESET']}")


def remove_readonly(func, path, excinfo):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def safe_subprocess(cmd: list) -> str:
    try:
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=10, text=True, **kwargs
        ).strip()
    except Exception:
        return ""


def resolve_hardware_matrix() -> tuple[str, str]:
    if sys.platform == "darwin":
        uname = safe_subprocess(["uname", "-m"])
        if "arm64" in uname:
            return "MPS", ""
        return "CPU", ""

    has_nvidia, has_amd, has_intel = False, False, False
    detected_amd_targets = []

    if os.name == "nt":
        gpu_info = safe_subprocess([
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
        ])
        for line in gpu_info.splitlines():
            name = line.strip()
            if not name:
                continue
            if re.search(r"(?i)NVIDIA", name):
                has_nvidia = True
            if re.search(r"(?i)Intel.*(Arc|Iris|Ultra)", name):
                has_intel = True
            if re.search(r"(?i)AMD|Radeon|Ryzen", name):
                has_amd = True
                if re.search(r"(?i)MI325|MI350", name):
                    detected_amd_targets.append("gfx950")
                elif re.search(r"(?i)MI300", name):
                    detected_amd_targets.append("gfx942")
                elif re.search(r"(?i)MI250", name):
                    detected_amd_targets.append("gfx90a")
                elif re.search(r"(?i)R9700|R9\d00|AI\s*PRO\s*R|890M|880M|Strix|Ryzen\s*AI|RX\s*[89]\d{2,3}", name):
                    detected_amd_targets.append("gfx120X")
                elif re.search(r"(?i)7900|W7900|7900M", name):
                    detected_amd_targets.append("gfx1100")
                elif re.search(r"(?i)7800|7700", name):
                    detected_amd_targets.append("gfx1101")
                elif re.search(r"(?i)7600|W7600|W7500", name):
                    detected_amd_targets.append("gfx1102")
                elif re.search(r"(?i)6900|6800|6700|W6800", name):
                    detected_amd_targets.append("gfx1030")
                elif re.search(r"(?i)780M|760M|740M|Phoenix|Hawk", name):
                    detected_amd_targets.append("gfx1103")
                elif re.search(r"(?i)680M|660M|Rembrandt", name):
                    detected_amd_targets.append("gfx1035")
                else:
                    detected_amd_targets.append("gfx1100")
    else:
        lspci = safe_subprocess(["lspci"])
        if lspci:
            if re.search(r"(?i)NVIDIA", lspci):
                has_nvidia = True
            if re.search(r"(?i)AMD|Radeon", lspci):
                has_amd = True
            if re.search(r"(?i)Intel.*(Arc|Graphics)", lspci):
                has_intel = True

        drm_path = Path("/sys/class/drm")
        if drm_path.exists():
            for uevent in drm_path.glob("card*/device/uevent"):
                try:
                    content = uevent.read_text()
                    if "DRIVER=amdgpu" in content:
                        has_amd = True
                    if "DRIVER=nvidia" in content:
                        has_nvidia = True
                    if "DRIVER=i915" in content or "DRIVER=xe" in content:
                        has_intel = True
                except Exception:
                    pass

        if has_amd:
            rocm_info = safe_subprocess(["rocminfo"])
            gfx_match = re.search(r"gfx\d+[a-zA-Z]?", rocm_info)
            if gfx_match:
                detected_amd_targets.append(gfx_match.group(0))
            elif lspci and re.search(r"(?i)MI325|MI350", lspci):
                detected_amd_targets.append("gfx950")
            elif lspci and re.search(r"(?i)890M|880M", lspci):
                detected_amd_targets.append("gfx120X")
            elif lspci and re.search(r"(?i)7800|7700", lspci):
                detected_amd_targets.append("gfx1101")
            elif lspci and re.search(r"(?i)7600|W7600|W7500", lspci):
                detected_amd_targets.append("gfx1102")
            elif lspci and re.search(r"(?i)680M|660M|Rembrandt", lspci):
                detected_amd_targets.append("gfx1035")
            else:
                detected_amd_targets.append("gfx1100")

    if has_nvidia:
        smi = safe_subprocess(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        if smi:
            try:
                major = int(smi.split(".")[0])
                if major >= 620:
                    return "CUDA_13_2", ""
                if major >= 600:
                    return "CUDA_13_0", ""
                if major >= 560:
                    return "CUDA_12_6", ""
                if major >= 520:
                    return "CUDA_12_1", ""
                if major >= 450:
                    return "CUDA_11_8", ""
            except Exception:
                pass
        return "CUDA_12_6", ""

    if has_amd:
        detected_amd_targets.sort(key=lambda x: AMD_GFX_RANK.get(x, 0), reverse=True)
        best_target = detected_amd_targets[0] if detected_amd_targets else "gfx1100"
        return "ROCM", best_target

    if has_intel:
        return "INTEL_XPU", ""

    return "CPU", ""


def fetch_runtime(version: str, target_dir: Path, isolated_env: dict) -> Path:
    if target_dir.exists():
        shutil.rmtree(target_dir, onerror=remove_readonly)
    target_dir.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        archive_name = f"python-{version}-embed-amd64.zip"
        url = f"https://www.python.org/ftp/python/{version}/{archive_name}"
        archive_path = target_dir.parent / archive_name
        print_status(f"Streaming standalone Windows runtime ({version})...")
        ctx = urllib.request.Request(url, headers={"User-Agent": "UniGen-Core"})
        with urllib.request.urlopen(ctx, timeout=30) as response, open(archive_path, "wb") as out_file:
            while chunk := response.read(65536):
                out_file.write(chunk)
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        archive_path.unlink()

        for pth_file in target_dir.glob("*._pth"):
            orig_lines = pth_file.read_text(encoding="utf-8").splitlines()
            core_zips = [line.strip() for line in orig_lines if line.strip().endswith(".zip")]
            payload_paths = core_zips + [".", "Lib/site-packages", "import site"]
            pth_file.write_text("\n".join(payload_paths) + "\n", encoding="utf-8")
        executable = target_dir / "python.exe"
    else:
        arch = "x86_64" if sys.maxsize > 2**32 else "i686"
        triple = f"{arch}-unknown-linux-gnu" if sys.platform.startswith("linux") else f"{arch}-apple-darwin"
        archive_name = f"cpython-{version}+{STANDALONE_RELEASE}-{triple}-install_only.tar.gz"
        url = f"https://github.com/astral-sh/python-build-standalone/releases/download/{STANDALONE_RELEASE}/{archive_name}"
        archive_path = target_dir.parent / archive_name
        print_status(f"Streaming standalone POSIX runtime ({version})...")
        ctx = urllib.request.Request(url, headers={"User-Agent": "UniGen-Core"})
        with urllib.request.urlopen(ctx, timeout=30) as response, open(archive_path, "wb") as out_file:
            while chunk := response.read(65536):
                out_file.write(chunk)
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(target_dir.parent)
        archive_path.unlink()

        source_extracted = target_dir.parent / "python"
        if source_extracted.exists() and source_extracted != target_dir:
            source_extracted.rename(target_dir)
        executable = target_dir / "bin" / "python"

    pip_bootstrapper = target_dir / "get-pip.py"
    v_parts = version.split(".")
    if len(v_parts) >= 2 and v_parts[0] == "3" and v_parts[1] in ["6", "7", "8", "9"]:
        pip_url = f"https://bootstrap.pypa.io/pip/{v_parts[0]}.{v_parts[1]}/get-pip.py"
    else:
        pip_url = "https://bootstrap.pypa.io/get-pip.py"

    pip_ctx = urllib.request.Request(pip_url, headers={"User-Agent": "UniGen-Core"})
    with urllib.request.urlopen(pip_ctx, timeout=30) as response, open(pip_bootstrapper, "wb") as out_file:
        while chunk := response.read(65536):
            out_file.write(chunk)

    subprocess.run(
        [str(executable), "-I", str(pip_bootstrapper), "--no-warn-script-location"],
        env=isolated_env,
        check=True
    )
    pip_bootstrapper.unlink()
    return executable


def condition_pytorch_runtime(executable: Path, isolated_env: dict) -> None:
    print_status("Resolving host accelerator matrix for PyTorch configuration...")
    profile, llvm_target = resolve_hardware_matrix()
    target_spec = f" ({llvm_target})" if llvm_target else ""
    print_status(f"Compute Architecture Locked: {profile}{target_spec}", "SUCCESS")

    pip_cmd = [str(executable), "-m", "pip", "install", "--no-cache-dir", "--no-warn-script-location"]

    print_status("Upgrading core build tooling (pip, setuptools, wheel)...")
    subprocess.run(
        pip_cmd + ["-U", "pip", "setuptools", "wheel"],
        env=isolated_env,
        check=True
    )

    print_status(f"Deploying hardware-aligned PyTorch binaries for {profile}...")
    if profile == "ROCM":
        url_param = f"{llvm_target}-all"
        index_url = f"https://rocm.nightlies.amd.com/v2-staging/{url_param}/"
        if os.name == "nt":
            subprocess.run(
                pip_cmd + ["--index-url", index_url, "--pre", "-U", "--no-build-isolation", "rocm[libraries,devel]"],
                env=isolated_env,
                check=True
            )
        subprocess.run(
            pip_cmd + ["--index-url", index_url, "--pre", "-U", "torch", "torchvision", "torchaudio"],
            env=isolated_env,
            check=True
        )
    elif profile.startswith("CUDA_"):
        cuda_urls = {
            "CUDA_13_2": "https://download.pytorch.org/whl/nightly/cu132",
            "CUDA_13_0": "https://download.pytorch.org/whl/nightly/cu130",
            "CUDA_12_6": "https://download.pytorch.org/whl/nightly/cu126",
            "CUDA_12_1": "https://download.pytorch.org/whl/cu121",
            "CUDA_11_8": "https://download.pytorch.org/whl/cu118",
        }
        index_url = cuda_urls.get(profile, "https://download.pytorch.org/whl/nightly/cu126")
        subprocess.run(
            pip_cmd + ["--index-url", index_url, "--pre", "-U", "torch", "torchvision", "torchaudio"],
            env=isolated_env,
            check=True
        )
    elif profile == "INTEL_XPU":
        subprocess.run(
            pip_cmd + [
                "--index-url",
                "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/",
                "-U",
                "torch",
                "torchvision",
                "torchaudio",
                "intel-extension-for-pytorch"
            ],
            env=isolated_env,
            check=True
        )
    elif profile == "MPS":
        subprocess.run(
            pip_cmd + ["-U", "torch", "torchvision", "torchaudio"],
            env=isolated_env,
            check=True
        )
    else:
        subprocess.run(
            pip_cmd + ["--index-url", "https://download.pytorch.org/whl/cpu", "--pre", "-U", "torch", "torchvision", "torchaudio"],
            env=isolated_env,
            check=True
        )

    print_status("PyTorch hardware alignment complete.", "SUCCESS")


def generate_environment_anchors(target_dir: Path, executable: Path) -> None:
    project_root = target_dir.parent
    rel_env_name = target_dir.name

    # Provision local hermetic MIOpen cache directories
    miopen_db = project_root / ".hf_cache" / "miopen" / "db"
    miopen_kernels = project_root / ".hf_cache" / "miopen" / "kernels"
    miopen_db.mkdir(parents=True, exist_ok=True)
    miopen_kernels.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        bat_content = f"""@echo off
title Hermetic Sandbox Environment
color 0b
set "ROOT_DIR=%~dp0"
set "PYTHONPATH="
set "PYTHONCASEOK="
set "VIRTUAL_ENV="
set "PYTHONIOENCODING=utf-8"
set "PYTHONHOME=%ROOT_DIR%{rel_env_name}"
set "PATH=%ROOT_DIR%{rel_env_name};%ROOT_DIR%{rel_env_name}\\Scripts;%PATH%"
set "HF_HOME=%ROOT_DIR%.hf_cache"
set "MIOPEN_USER_DB_PATH=%ROOT_DIR%.hf_cache\\miopen\\db"
set "MIOPEN_CUSTOM_CACHE_DIR=%ROOT_DIR%.hf_cache\\miopen\\kernels"
set "MIOPEN_FIND_MODE=2"
set "MIOPEN_LOG_LEVEL=0"
set "MIOPEN_ENABLE_LOGGING=0"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo =======================================================================
echo  Hermetic Sandbox Shell Active
echo  Interpreter: %PYTHONHOME%\\python.exe
echo  Cache Anchor: %HF_HOME%
echo  MIOpen Cache: %MIOPEN_CUSTOM_CACHE_DIR% (Find Mode: FAST Heuristic)
echo  Isolation Status: Host PYTHONPATH cleared. Anchored relative root.
echo =======================================================================
cmd /k
"""
        launcher_file = project_root / "launch_env.bat"
        launcher_file.write_text(bat_content, encoding="utf-8")
    else:
        sh_content = f"""#!/usr/bin/env bash
ROOT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" &> /dev/null && pwd)"
unset PYTHONPATH
unset PYTHONCASEOK
unset VIRTUAL_ENV
export PYTHONIOENCODING="utf-8"
export PYTHONHOME="${{ROOT_DIR}}/{rel_env_name}"
export PATH="${{ROOT_DIR}}/{rel_env_name}:${{ROOT_DIR}}/{rel_env_name}/bin:${{PATH}}"
export HF_HOME="${{ROOT_DIR}}/.hf_cache"
export MIOPEN_USER_DB_PATH="${{ROOT_DIR}}/.hf_cache/miopen/db"
export MIOPEN_CUSTOM_CACHE_DIR="${{ROOT_DIR}}/.hf_cache/miopen/kernels"
export MIOPEN_FIND_MODE="2"
export MIOPEN_LOG_LEVEL="0"
export MIOPEN_ENABLE_LOGGING="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
echo "======================================================================="
echo " Hermetic Sandbox Shell Active"
echo " Interpreter: ${{PYTHONHOME}}/bin/python"
echo " Cache Anchor: ${{HF_HOME}}"
echo " MIOpen Cache: ${{MIOPEN_CUSTOM_CACHE_DIR}} (Find Mode: FAST Heuristic)"
echo " Isolation Status: Host modules segregated. Anchored relative root."
echo "======================================================================="
exec $SHELL
"""
        launcher_file = project_root / "launch_env.sh"
        launcher_file.write_text(sh_content, encoding="utf-8")
        launcher_file.chmod(0o755)

    print_status(f"Portable dynamic environment launcher deployed: {launcher_file.name}", "SUCCESS")


def main() -> None:
    print_status("======================================================")
    print_status("            INTELIVENV ENVIRONMENT CONDITIONER        ", "SUCCESS")
    print_status("======================================================")

    for key, version in PREDEFINED_MATRIX.items():
        print(f"  [{key}] Python {version}")

    manual_option = str(len(PREDEFINED_MATRIX) + 1)
    print(f"  [{manual_option}] Manual Version Entry")

    choice = input("\nSelect core runtime configuration: ").strip()
    if choice in PREDEFINED_MATRIX:
        selected_version = PREDEFINED_MATRIX[choice]
    elif choice == manual_option:
        selected_version = input("Enter custom Python version (e.g., 3.10.11, 3.11.9): ").strip()
    else:
        print_status("Invalid menu selection. Exiting.", "ERROR")
        sys.exit(1)

    if not selected_version:
        print_status("Execution terminated: No valid version targets identified.", "ERROR")
        sys.exit(1)

    project_root = Path(__file__).resolve().parent
    target_sandbox = project_root / f"py_env_{selected_version.replace('.', '_')}"

    isolated_env = os.environ.copy()
    for var in ["PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONCASEOK"]:
        isolated_env.pop(var, None)
    isolated_env["PYTHONIOENCODING"] = "utf-8"

    try:
        executable = fetch_runtime(selected_version, target_sandbox, isolated_env)
        condition_pytorch_runtime(executable, isolated_env)
        generate_environment_anchors(target_sandbox, executable)
        print_status(
            f"Hermetic environment conditioned successfully for Python {selected_version}.",
            "SUCCESS"
        )
    except Exception as e:
        print_status(f"Environment conditioning failed: {str(e)}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()