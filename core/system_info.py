"""Real hardware and toolchain reporting.

Replaces the previous `gpu_accelerator.report_acceleration_mode()`, which
printed "12 GB VRAM" and "32-thread FFT" as string literals regardless of what
the machine actually had.
"""
import os
import platform
import shutil
import subprocess

import numpy as np


def cpu_model() -> str:
    """Best-effort CPU model name, portable across Linux/WSL, macOS, Windows."""
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except OSError:
        pass

    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass

    return platform.processor() or platform.machine() or "unknown CPU"


def cpu_count() -> int:
    """Cores actually usable by this process, honouring cgroup/affinity limits."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def blas_backend() -> str:
    try:
        config = np.__config__.show(mode="dicts")
        build = config.get("Build Dependencies", {}).get("blas", {})
        name = build.get("name")
        if name:
            return str(name)
    except Exception:
        pass
    return "unknown"


def cuda_info():
    """Returns (device_name, total_vram_bytes) or None. Never raises."""
    try:
        import torch
    except Exception:
        # ImportError, but also OSError from a half-installed CUDA runtime.
        return None
    try:
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return (props.name, props.total_memory)
    except Exception:
        return None


def find_tool(name: str) -> str:
    return shutil.which(name) or ""


def describe_tools() -> dict:
    """Presence and version of every external binary the indexer shells out to."""
    tools = {}
    for name, version_args in (
        ("ffmpeg", ["-version"]),
        ("yt-dlp", ["--version"]),
        ("sqlite3", ["--version"]),
        ("adb", ["version"]),
    ):
        path = find_tool(name)
        version = ""
        if path:
            try:
                out = subprocess.run(
                    [path] + version_args, capture_output=True, text=True, timeout=10
                )
                if out.returncode == 0:
                    version = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
            except (OSError, subprocess.SubprocessError):
                version = "(version check failed)"
        tools[name] = {"path": path, "version": version}
    return tools


def report(stream=None) -> None:
    """Prints what this machine really is."""
    write = (stream.write if stream else print)

    def line(text=""):
        if stream:
            stream.write(text + "\n")
        else:
            print(text)

    cores = cpu_count()
    line(f"[HOST]  {platform.system()} {platform.release()} | Python {platform.python_version()}")
    line(f"[CPU]   {cpu_model()} | {cores} usable core(s)")
    line(f"[NUMPY] {np.__version__} | BLAS: {blas_backend()}")

    cuda = cuda_info()
    if cuda:
        name, vram = cuda
        line(f"[CUDA]  {name} | {vram / (1024 ** 3):.1f} GiB VRAM (unused: DSP is CPU-bound)")
    else:
        line("[CUDA]  not available (not needed: the FFT work is ~0.05s/track)")

    for name, info in describe_tools().items():
        if info["path"]:
            line(f"[TOOL]  {name}: {info['path']} {info['version']}")
        else:
            line(f"[TOOL]  {name}: NOT FOUND")
