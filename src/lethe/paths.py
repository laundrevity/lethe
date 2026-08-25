"""Project-relative paths. Everything lives under the repo root, never in ~/.cache."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("LETHE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # src/lethe/paths.py -> repo root
    return Path(__file__).resolve().parents[2]


def models_dir() -> Path:
    d = project_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_dir() -> Path:
    d = project_root() / ".lethe"
    d.mkdir(parents=True, exist_ok=True)
    return d


def llama_dir() -> Path:
    return project_root() / "vendor" / "llama.cpp"


def llama_bin(name: str) -> Path:
    return llama_dir() / "build" / "bin" / name


def repo_dirname(repo_id: str) -> str:
    """HauhauCS/Foo-GGUF -> HauhauCS--Foo-GGUF"""
    return repo_id.replace("/", "--")


def find_ggufs() -> list[Path]:
    return sorted(p for p in models_dir().rglob("*.gguf") if p.is_file())
