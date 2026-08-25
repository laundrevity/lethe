"""lethe.toml at the project root: configuration only, never content. Gitignored."""

from __future__ import annotations

import tomllib
from pathlib import Path

from .paths import project_root


def config_path() -> Path:
    return project_root() / "lethe.toml"


def load() -> dict:
    p = config_path()
    if not p.exists():
        return {}
    try:
        return tomllib.loads(p.read_text())
    except Exception:
        return {}


def default_model() -> str | None:
    return load().get("default", {}).get("model")


def set_default_model(spec: str) -> None:
    cfg = load()
    cfg.setdefault("default", {})["model"] = spec
    lines = ["# lethe configuration — no conversation content ever lives here", "", "[default]"]
    lines.append(f'model = "{cfg["default"]["model"]}"')
    config_path().write_text("\n".join(lines) + "\n")
