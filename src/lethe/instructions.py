"""LETHE.md: the user's standing instructions, appended to the system prompt. Reads only; writes nothing.

Two layers, both optional:
- global:  <lethe repo>/LETHE.md — applies to every chat, wherever it is launched.
- project: the nearest LETHE.md (else AGENTS.md) from the launch directory upward, stopping at the
  repository root or the home directory.
"""

from __future__ import annotations

from pathlib import Path

from .paths import project_root

NAMES = ("LETHE.md", "AGENTS.md")
MAX_CHARS = 32_000  # per file, ≈8K tokens
MARKER = "\n\n# Standing instructions from the user\n"
PREAMBLE = (
    "The user wrote the following instruction files for you. They apply to this whole conversation; follow them. "
    "Project instructions take precedence over global ones where they conflict.\n"
)


def find_project_file(start: Path) -> Path | None:
    home = Path.home().resolve()
    d = start.expanduser().resolve()
    for d in (d, *d.parents):
        if d == home or d == d.parent:
            return None
        for name in NAMES:
            if (d / name).is_file():
                return d / name
        if (d / ".git").exists():
            return None
    return None


def _read(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + f"\n[…truncated {len(text) - MAX_CHARS:,} chars]"
    return text


def load(start: Path) -> tuple[str, list[tuple[str, Path, int]]]:
    """(block to append to the system prompt, [(layer, path, chars)]) — block is "" when nothing is found."""
    found: list[tuple[str, Path]] = []
    global_file = project_root() / NAMES[0]
    if global_file.is_file():
        found.append(("global", global_file))
    project_file = find_project_file(start)
    if project_file is not None and project_file.resolve() != global_file.resolve():
        found.append(("project", project_file))
    sections, loaded = [], []
    for layer, path in found:
        try:
            text = _read(path)
        except OSError:
            continue
        if text:
            sections.append(f"\n## {layer.capitalize()} instructions ({path})\n\n{text}\n")
            loaded.append((layer, path, len(text)))
    return (MARKER + PREAMBLE + "".join(sections) if sections else ""), loaded


def strip(text: str) -> str:
    return text.partition(MARKER)[0]


def apply(text: str, block: str) -> str:
    """Replace any earlier instruction block (e.g. in a resumed conversation) with the current one."""
    return strip(text) + block
