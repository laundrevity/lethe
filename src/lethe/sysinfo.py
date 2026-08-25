"""Machine facts that decide what fits: RAM, GPU-wired limit, cores."""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass

GiB = 1024**3


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def _metal_device_bytes() -> int | None:
    """Parse `llama-server --list-devices`: 'MTL0: Apple M5 Max (110100 MiB, 110099 MiB free)'."""
    from .paths import llama_bin

    exe = llama_bin("llama-server")
    if not exe.exists():
        return None
    try:
        out = subprocess.run([str(exe), "--list-devices"], capture_output=True, text=True, timeout=15)
    except Exception:
        return None
    m = re.search(r"MTL\d+: .*?\((\d+) MiB", out.stdout + out.stderr)
    return int(m.group(1)) * 1024 * 1024 if m else None


@dataclass
class Machine:
    total_ram: int  # bytes
    gpu_wired_limit: int  # bytes the GPU may wire; the real ceiling for -ngl all
    ncpu: int
    chip: str
    is_apple_silicon: bool

    @property
    def usable_for_model(self) -> int:
        """Budget we hand to a model: GPU-wired ceiling minus ~4 GiB headroom for OS + compute buffers."""
        return max(0, self.gpu_wired_limit - 4 * GiB)


def machine() -> Machine:
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
