"""Machine facts that decide what fits: RAM, GPU-wired limit, cores."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import psutil

GiB = 1024**3


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def _gpu_device_memory() -> tuple[int, int] | None:
    """Parse `llama-server --list-devices`: 'MTL0: Apple M5 Max (110100 MiB, 110099 MiB free)'."""
    from .paths import llama_bin

    exe = llama_bin("llama-server")
    if not exe.exists():
        return None
    try:
        out = subprocess.run([str(exe), "--list-devices"], capture_output=True, text=True, timeout=15)
    except Exception:
        return None
    m = re.search(r"(?:MTL|Vulkan|ROCm|HIP|CUDA)\d+: .*?\((\d+) MiB, (\d+) MiB free\)", out.stdout + out.stderr)
    return (int(m.group(1)) * 1024**2, int(m.group(2)) * 1024**2) if m else None


def _metal_device_bytes() -> int | None:
    memory = _gpu_device_memory()
    return memory[0] if memory else None


@dataclass
class OtherProcess:
    pid: int
    rss: int
    cmd: str


def wired_bytes() -> int:
    """Memory currently wired down system-wide (GPU-resident model weights + KV show up here)."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
        page = int(re.search(r"page size of (\d+)", out).group(1))
        wired = int(re.search(r"Pages wired down:\s+(\d+)", out).group(1))
        return wired * page
    except Exception:
        return 0


def llama_processes() -> list[OtherProcess]:
    """Every llama-server / llama-cli on the machine, ours or not, with resident memory."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,rss=,args="], capture_output=True, text=True, check=True).stdout
    except Exception:
        return []
    procs = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, rss_kb, cmd = parts
        exe = cmd.split()[0]
        if exe.endswith(("llama-server", "llama-cli", "llama-bench")):
            procs.append(OtherProcess(int(pid), int(rss_kb) * 1024, cmd))
    return procs


@dataclass
class Machine:
    total_ram: int  # bytes
    gpu_wired_limit: int  # bytes the GPU may wire; the real ceiling for -ngl all
    ncpu: int
    chip: str
    is_apple_silicon: bool
    gpu_free: int | None = None

    @property
    def usable_for_model(self) -> int:
        """Capacity we hand to a model: GPU-wired ceiling minus ~4 GiB headroom for OS + compute buffers."""
        return max(0, self.gpu_wired_limit - 4 * GiB)

    def available_now(self, excluding_rss: int = 0) -> int:
        """What is free *right now*: capacity minus what's already wired by everything else.
        `excluding_rss` = memory of our own server that is about to be stopped."""
        if platform.system() == "Linux":
            # Both host RAM and the GPU allocation budget constrain UMA. Do not
            # subtract resident memory twice from the driver's free budget.
            available = max(0, psutil.virtual_memory().available + excluding_rss - 4 * GiB)
            gpu = self.gpu_wired_limit if self.gpu_free is None else self.gpu_free + excluding_rss
            return min(self.usable_for_model, available, max(0, gpu - 4 * GiB))
        committed = max(0, wired_bytes() - excluding_rss)
        return max(0, self.usable_for_model - committed)


def machine() -> Machine:
    if platform.system() == "Linux":
        total = psutil.virtual_memory().total
        chip = platform.processor() or "unknown"
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    chip = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
        memory = _gpu_device_memory()
        # No detected GPU means no verified GPU budget, not all host RAM.
        limit, free = memory if memory else (0, 0)
        return Machine(total, min(total, limit), os.cpu_count() or 1, chip, False, free)
    total = int(_sysctl("hw.memsize") or 0)
    ncpu = int(_sysctl("hw.ncpu") or 1)
    chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown"
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"

    # Best source: what the Metal backend itself reports (recommendedMaxWorkingSetSize).
    # Fallback: iogpu.wired_limit_mb (0 = macOS default, ~75% of RAM on big machines).
    wired = _metal_device_bytes()
    if wired is None:
        wired_mb = int(_sysctl("iogpu.wired_limit_mb") or 0)
        if wired_mb > 0:
            wired = wired_mb * 1024 * 1024
        elif apple:
            wired = int(total * 0.75) if total >= 36 * GiB else int(total * 2 / 3)
        else:
            wired = total

    return Machine(total_ram=total, gpu_wired_limit=wired, ncpu=ncpu, chip=chip, is_apple_silicon=apple)


def fmt_bytes(n: float) -> str:
    if n >= GiB:
        return f"{n / GiB:.1f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.0f} MiB"
    return f"{n / 1024:.0f} KiB"
