"""Model identifiers.

The canonical form is `owner/repo:QUANT`, e.g.
    HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:Q8_K_P
Also accepted: a path to a .gguf, `owner/repo` (if the local dir holds exactly one model),
or any unique substring of a local file name (`q8_k_p`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .ggufmeta import quant_from_filename
from .hub import SHARD_RE
from .paths import find_ggufs, models_dir, repo_dirname

SIDECAR_RE = re.compile(r"^(mmproj|imatrix|mtp-)|fastmtp|-draft", re.I)


def is_sidecar(name: str) -> bool:
    return bool(SIDECAR_RE.search(name))


@dataclass(frozen=True)
class ModelSpec:
    raw: str
    repo: str | None = None
    quant: str | None = None
    path: Path | None = None

    @classmethod
    def parse(cls, s: str) -> "ModelSpec":
        s = s.strip()
        p = Path(s).expanduser()
        if s.endswith(".gguf") and p.is_file():
            return cls(raw=s, path=p.resolve())
        head, sep, tail = s.partition(":")
        if "/" in head and not head.startswith((".", "/", "~")):
            return cls(raw=s, repo=head, quant=(tail.upper() or None) if sep else None)
        return cls(raw=s)

    @property
    def is_hub(self) -> bool:
        return self.repo is not None

    def __str__(self) -> str:
        if self.repo:
            return f"{self.repo}:{self.quant}" if self.quant else self.repo
        return self.raw


@dataclass
class LocalModel:
    path: Path  # first shard
    files: list[Path]  # all shards
    size: int
    repo: str | None
    quant: str | None

    @property
    def spec(self) -> str:
        if self.repo and self.quant:
            return f"{self.repo}:{self.quant}"
        return str(self.path.relative_to(models_dir())) if self.path.is_relative_to(models_dir()) else str(self.path)

    @property
    def is_sidecar(self) -> bool:
        return is_sidecar(self.path.name)


def _repo_from_dir(d: Path) -> str | None:
    """models/<owner>--<repo>/ -> owner/repo"""
    try:
        rel = d.relative_to(models_dir())
    except ValueError:
        return None
    top = rel.parts[0] if rel.parts else ""
    return top.replace("--", "/", 1) if "--" in top else None


def local_models(include_sidecars: bool = False) -> list[LocalModel]:
    groups: dict[Path, list[Path]] = {}
    for p in find_ggufs():
        m = SHARD_RE.match(p.name)
        key = p.with_name(m["base"] + ".gguf") if m else p
        groups.setdefault(key, []).append(p)
    out = []
    for key, files in groups.items():
        files.sort()
        lm = LocalModel(
            path=files[0],
            files=files,
            size=sum(f.stat().st_size for f in files),
            repo=_repo_from_dir(files[0].parent),
            quant=quant_from_filename(key.name),
        )
        if include_sidecars or not lm.is_sidecar:
            out.append(lm)
    return sorted(out, key=lambda x: x.size)


class ModelNotFound(Exception):
    def __init__(self, spec: ModelSpec, msg: str, candidates: list[LocalModel] | None = None):
        super().__init__(msg)
        self.spec, self.candidates = spec, candidates or []


def resolve(spec: ModelSpec | str) -> LocalModel:
    """Find a downloaded model for the spec. Raises ModelNotFound (with .spec) if it isn't local."""
    if isinstance(spec, str):
        spec = ModelSpec.parse(spec)
    models = local_models()
    if spec.path:
        for lm in models:
            if spec.path in lm.files:
                return lm
        return LocalModel(path=spec.path, files=[spec.path], size=spec.path.stat().st_size, repo=None, quant=quant_from_filename(spec.path.name))
    if spec.repo:
        in_repo = [lm for lm in models if lm.repo and lm.repo.lower() == spec.repo.lower()]
        if spec.quant:
            hits = [lm for lm in in_repo if (lm.quant or "").upper() == spec.quant]
            if len(hits) == 1:
                return hits[0]
            raise ModelNotFound(spec, f"{spec} is not downloaded" + (f"; local quants: {', '.join(lm.quant or '?' for lm in in_repo)}" if in_repo else ""), in_repo)
        if len(in_repo) == 1:
            return in_repo[0]
        if not in_repo:
            raise ModelNotFound(spec, f"nothing from {spec.repo} is downloaded")
        raise ModelNotFound(spec, f"{spec.repo} has several local quants — say which: " + ", ".join(lm.spec for lm in in_repo), in_repo)
    # substring of the relative path
    needle = spec.raw.lower()
    hits = [lm for lm in models if needle in str(lm.path.relative_to(models_dir())).lower() or needle in lm.spec.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ModelNotFound(spec, f"no local model matches '{spec.raw}'")
    raise ModelNotFound(spec, f"'{spec.raw}' is ambiguous: " + ", ".join(lm.spec for lm in hits), hits)


def local_dir_name(repo: str) -> str:
    return repo_dirname(repo)
