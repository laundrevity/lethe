"""Hugging Face Hub: list a repo's GGUFs (with remote header parsing) and download into ./models."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# Never send usage telemetry to the Hub. Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url  # noqa: E402
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, build_hf_headers  # noqa: E402

from .ggufmeta import ModelShape, NeedMoreData, parse_remote, shape_from_kv
from .paths import models_dir, repo_dirname

SHARD_RE = re.compile(r"^(?P<base>.+?)-(?P<idx>\d{5})-of-(?P<n>\d{5})\.gguf$")


@dataclass
class HubGguf:
    """One logical model in a repo: a single .gguf or a group of shards."""

    repo_id: str
    base: str  # display/base name
    files: list[str]  # actual rfilenames (1 or N shards)
    size: int  # total bytes
    shape: ModelShape | None = None
    error: str | None = None

    @property
    def first_file(self) -> str:
        return self.files[0]

    @property
    def is_sidecar(self) -> bool:
        b = self.base.lower()
        return b.startswith("mmproj") or "mtp" in b and "draft" in b or "fastmtp" in b


def list_repo_ggufs(repo_id: str, revision: str = "main") -> list[HubGguf]:
    api = HfApi()
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    groups: dict[str, HubGguf] = {}
    for s in info.siblings or []:
        fn = s.rfilename
        if not fn.lower().endswith(".gguf"):
            continue
        size = s.size or (s.lfs.size if s.lfs else 0) or 0
        m = SHARD_RE.match(Path(fn).name)
        if m:
            key = str(Path(fn).with_name(m["base"] + ".gguf"))
        else:
            key = fn
        g = groups.get(key)
        if g is None:
            g = groups[key] = HubGguf(repo_id=repo_id, base=Path(key).name, files=[], size=0)
        g.files.append(fn)
        g.size += size
    for g in groups.values():
        g.files.sort()
    return sorted(groups.values(), key=lambda g: g.size)


def fetch_shape(g: HubGguf, revision: str = "main") -> HubGguf:
    """Parse the GGUF header of the first shard over HTTP Range. Mutates and returns g."""
    try:
        url = hf_hub_url(g.repo_id, g.first_file, revision=revision)
        kv = parse_remote(url, headers=build_hf_headers())
        g.shape = shape_from_kv(kv, g.base)
    except NeedMoreData as e:
        g.error = f"metadata too large: {e}"
    except Exception as e:  # noqa: BLE001
        g.error = f"{type(e).__name__}: {e}"
    return g


def local_dir_for(repo_id: str) -> Path:
    return models_dir() / repo_dirname(repo_id)


def download(repo_id: str, filenames: list[str], revision: str = "main", anonymous: bool = True) -> list[Path]:
    """Download files into models/<owner>--<repo>/ (no symlinks into ~/.cache).

    Anonymous by default so public downloads aren't tied to the HF account behind HF_TOKEN;
    falls back to the token automatically for gated/private repos.
    """
    out = local_dir_for(repo_id)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for fn in filenames:
        try:
            p = hf_hub_download(repo_id, fn, revision=revision, local_dir=out, token=False if anonymous else None)
        except (GatedRepoError, HfHubHTTPError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if not anonymous or status not in (401, 403):
                raise
            p = hf_hub_download(repo_id, fn, revision=revision, local_dir=out)
        paths.append(Path(p))
    return paths


def match_quant(ggufs: list[HubGguf], quant: str) -> list[HubGguf]:
    q = quant.upper()
    return [g for g in ggufs if re.search(rf"(?<![A-Z0-9_]){re.escape(q)}(?![A-Z0-9_])", g.base.upper())]
