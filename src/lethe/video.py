"""Local video -> timestamped PNG frames, in memory only; no audio or network.

Do not send input_video to llama.cpp: its FFmpeg cache:pipe input can create a
plaintext temporary file. Read the user's existing file and pipe PNGs to Python.
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from .attachments import UnsupportedFile


def _run(program: str, args: list[str], path: Path) -> bytes:
    executable = shutil.which(program)
    if not executable:
        raise UnsupportedFile(f"{program} is required for video/WebP attachments; install with `brew install ffmpeg`")
    try:
        result = subprocess.run(
            [executable, "-v", "error", *args], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True,
            # In particular, do not inherit FFREPORT, which writes a log file.
            env={"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired as e:
        raise UnsupportedFile(f"{path.name}: {program} timed out decoding media") from e
    except (OSError, subprocess.CalledProcessError) as e:
        raise UnsupportedFile(f"{path.name}: {program} could not decode this file") from e
    return result.stdout


def image_png(path: Path, *, timestamp: float | None = None) -> bytes:
    """Read one frame from an existing file; the only output is a PNG on stdout."""
    args = ["-nostdin", "-protocol_whitelist", "file,pipe", "-threads", "2"]
    if timestamp is not None:
        args += ["-ss", f"{timestamp:.6f}"]
    args += ["-i", str(path.resolve()), "-map", "0:v:0", "-an", "-sn", "-dn"]
    if timestamp is not None:
        args += ["-vf", "scale=w='min(768,iw)':h='min(768,ih)':force_original_aspect_ratio=decrease"]
    args += ["-frames:v", "1", "-threads", "1", "-c:v", "png", "-f", "image2pipe", "pipe:1"]
    data = _run("ffmpeg", args, path)
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise UnsupportedFile(f"{path.name}: no image frame could be decoded")
    return data


def video_parts(path: Path, max_frames: int = 16) -> list[dict]:
    """Sample the whole clip at <=1 fps, bounded by max_frames and 768px/edge.

    Timestamps are seek positions relative to the start of the video. Long
    clips are sampled more sparsely; the prompt explicitly describes the gaps.
    """
    if not 1 <= max_frames <= 128:
        raise UnsupportedFile("video frame limit must be between 1 and 128")
    raw = _run("ffprobe", [
        "-protocol_whitelist", "file,pipe", "-select_streams", "v:0",
        "-show_entries", "stream=duration:format=duration", "-of", "json", str(path.resolve()),
    ], path)
    try:
        info = json.loads(raw)
        stream = info["streams"][0]
        duration = float(stream.get("duration", info.get("format", {}).get("duration", "nan")))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid duration")
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise UnsupportedFile(f"{path.name}: could not determine a finite video duration") from e
    count = min(max_frames, math.ceil(duration))
    step = max(1.0, duration / max_frames)
    parts = [{"type": "text", "text": (
        f"Video {path.name!r}: {duration:.3f} seconds. "
        f"{count} sampled frames in chronological order, approximately {step:.3f}s apart. "
        "Only these frames are provided; events between them may be missed. No audio is included."
    )}]
    for i in range(count):
        timestamp = i * step
        data = image_png(path, timestamp=timestamp)
        parts += [
            {"type": "text", "text": f"Video frame at {timestamp:.3f}s:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}},
        ]
    return parts
