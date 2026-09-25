"""Minimal GGUF header/KV parser + memory math.

We parse only the metadata block (never tensor data), from either a local file or the first
few MiB of a remote HF file via HTTP Range. Large arrays (tokenizer vocab, merges) are skipped
and recorded by length only.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

GGUF_MAGIC = b"GGUF"
_SCALAR = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
    6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
T_STRING, T_ARRAY = 8, 9
MAX_ARRAY_KEEP = 512  # keep small arrays (per-layer head counts etc.), summarize big ones


class NeedMoreData(Exception):
    pass


class _Cursor:
    def __init__(self, buf: bytes):
        self.buf, self.pos = buf, 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise NeedMoreData
        b = self.buf[self.pos : self.pos + n]
        self.pos += n
        return b

    def scalar(self, t: int) -> Any:
        fmt, size = _SCALAR[t]
        return struct.unpack(fmt, self.take(size))[0]

    def string(self) -> str:
        n = self.scalar(10)
        return self.take(n).decode("utf-8", errors="replace")

    def value(self, t: int) -> Any:
        if t in _SCALAR:
            return self.scalar(t)
        if t == T_STRING:
            return self.string()
        if t == T_ARRAY:
            et = self.scalar(4)
            n = self.scalar(10)
            if n <= MAX_ARRAY_KEEP:
                return [self.value(et) for _ in range(n)]
            # skip big arrays cheaply
            if et in _SCALAR:
                self.take(n * _SCALAR[et][1])
            else:
                for _ in range(n):
                    self.value(et)
            return {"__array_len__": n}
        raise ValueError(f"unknown gguf type {t}")


def parse_kv(buf: bytes) -> dict[str, Any]:
    c = _Cursor(buf)
    if c.take(4) != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    version = c.scalar(4)
    n_tensors = c.scalar(10)
    n_kv = c.scalar(10)
    kv: dict[str, Any] = {"__version__": version, "__n_tensors__": n_tensors}
    for _ in range(n_kv):
        k = c.string()
        t = c.scalar(4)
        kv[k] = c.value(t)
    return kv


def parse_local(path: Path) -> dict[str, Any]:
    size = 1 << 22
    with open(path, "rb") as f:
        while True:
            f.seek(0)
            buf = f.read(size)
            try:
                return parse_kv(buf)
            except NeedMoreData:
                if size > path.stat().st_size:
                    raise
                size *= 4


def parse_remote(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    size = 1 << 24  # 16 MiB covers ~250K-token vocabs
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        for _ in range(4):
            r = client.get(url, headers={**(headers or {}), "Range": f"bytes=0-{size - 1}"})
            r.raise_for_status()
            try:
                return parse_kv(r.content)
            except NeedMoreData:
                size *= 4
    raise NeedMoreData("gguf metadata larger than 1 GiB?!")


# --------------------------------------------------------------------------- model shape


@dataclass
class ModelShape:
    arch: str
    name: str
    n_ctx_train: int
    n_layer: int
    n_attn_layers: int  # layers that keep a KV cache (== n_layer for dense transformers)
    n_head_kv: int
    key_len: int
    val_len: int
    n_embd: int
    n_vocab: int
    file_type: int | None
    quant: str | None
    params_b: float | None
    has_mtp: bool
    extra: dict[str, Any] = field(default_factory=dict)
    # sliding-window attention (Gemma-style iSWA): window layers cap their KV at swa_window tokens
    kv_elems_growing: int = 0  # per-token KV elements in layers whose cache grows with context
    kv_elems_swa: int = 0  # per-token KV elements in sliding-window layers
    swa_window: int = 0
    n_swa_layers: int = 0

    def kv_bytes_per_token(self, kv_bytes: float = 2) -> int:
        """Marginal cost of one more context token (F16 = 2 bytes/elem; q8_0 ≈ 1.06). SWA layers excluded."""
        return int(self.kv_elems_growing * kv_bytes)

    def kv_fixed_bytes(self, kv_bytes: float = 2) -> int:
        """Upper bound of the sliding-window layers' cache (they stop growing at swa_window tokens)."""
        return int(self.kv_elems_swa * self.swa_window * kv_bytes)

    def kv_bytes(self, n_ctx: int, kv_bytes: float = 2) -> int:
        return self.kv_bytes_per_token(kv_bytes) * n_ctx + int(self.kv_elems_swa * min(n_ctx, self.swa_window) * kv_bytes)

    def max_ctx_for(self, budget_bytes: int, model_bytes: int, kv_bytes: float = 2) -> int:
        per = self.kv_bytes_per_token(kv_bytes)
        free = budget_bytes - model_bytes - self.kv_fixed_bytes(kv_bytes)
        if per <= 0:
            return self.n_ctx_train if free >= 0 else 0
        if free <= 0:
            return 0
        return int(free // per)


# llama.cpp file_type enum -> human name (subset that matters)
FTYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
    11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
    17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4",
}


_QUANT_RE = re.compile(r"[-_.]((?:I?Q\d(?:_[A-Z0-9]+)*)|BF16|F16|F32|MXFP4)(?=\.gguf$|[-_.])", re.I)


def quant_from_filename(filename: str) -> str | None:
    """Vendor-specific quant names (Q8_K_P, UD-IQ4_XS…) only exist in filenames."""
    m = _QUANT_RE.search(filename)
    return m.group(1).upper() if m else None


def _get(kv: dict, key: str, default=None):
    v = kv.get(key, default)
    if isinstance(v, dict) and "__array_len__" in v:
        return default
    return v


# Sliding-window layer periods llama.cpp hard-codes when the GGUF carries no `sliding_window_pattern`
# (mirrors hparams.set_swa_pattern(period, dense_first) in vendor/llama.cpp/src/models/*.cpp).
# period p, dense_first=False: layers with (i+1) % p == 0 are full attention (e.g. gpt-oss: odd layers).
# dense_first=True: layers with i % p == 0 are full. p=0: every layer windowed; p=1: none.
SWA_PATTERN_DEFAULTS: dict[str, tuple[int, bool]] = {
    "gpt-oss": (2, False),
    "gemma2": (2, False),
    "gemma3": (6, False),
    "gemma3n": (5, False),
    "gemma-embedding": (6, False),
    "cohere2": (4, False),
    "llama4": (4, False),
    "olmo2": (4, False),
    "exaone4": (4, False),
    "exaone-moe": (4, False),
    "afmoe": (4, False),
    "mellum": (4, False),
    "muse-glimmer": (4, False),
    "plamo3": (8, False),
    "modern-bert": (3, True),
    "smallthinker": (4, True),
    "laguna": (4, True),
    "deepseek4": (0, False),
    "dflash": (0, False),
    "phi3": (1, False),
}


def _swa_period_layers(n_layer: int, period: int, dense_first: bool = False) -> list[bool]:
    if period == 0:
        return [True] * n_layer
    if dense_first:
        return [(i % period) != 0 for i in range(n_layer)]
    return [((i + 1) % period) != 0 for i in range(n_layer)]


def shape_from_kv(kv: dict[str, Any], filename: str = "") -> ModelShape:
    arch = kv.get("general.architecture", "unknown")
    a = lambda k, d=None: _get(kv, f"{arch}.{k}", d)  # noqa: E731

    n_layer = int(a("block_count", 0))
    n_embd = int(a("embedding_length", 0))
    n_head = a("attention.head_count", 0)
    n_head_kv_raw = kv.get(f"{arch}.attention.head_count_kv", n_head)  # keep per-layer arrays
    n_head_kv = n_head_kv_raw
    if isinstance(n_head_kv, list):
        n_head_kv = max(n_head_kv) if n_head_kv else 0
    if isinstance(n_head, list):
        n_head = max(n_head) if n_head else 0
    n_head, n_head_kv = int(n_head or 0), int(n_head_kv or 0)
    key_len = int(a("attention.key_length", (n_embd // n_head) if n_head else 0))
    val_len = int(a("attention.value_length", key_len))

    # sliding-window attention (Gemma-style): pattern is a per-layer bool array (True = window layer)
    # or an integer period p (every p-th layer global, like llama.cpp set_swa_pattern)
    swa_window = int(a("attention.sliding_window", 0) or 0)
    pattern = kv.get(f"{arch}.attention.sliding_window_pattern")
    key_swa = int(a("attention.key_length_swa", key_len) or key_len)
    val_swa = int(a("attention.value_length_swa", val_len) or val_len)
    if swa_window and pattern is not None and not (isinstance(pattern, dict)):
        if isinstance(pattern, list):
            is_swa = [bool(x) for x in pattern] + [False] * (n_layer - len(pattern))
        else:
            is_swa = _swa_period_layers(n_layer, int(pattern))
    elif swa_window and arch in SWA_PATTERN_DEFAULTS:
        # llama.cpp hard-codes the period for these archs (src/models/<arch>.cpp); the GGUF has no pattern key
        is_swa = _swa_period_layers(n_layer, *SWA_PATTERN_DEFAULTS[arch])
    elif swa_window:
        is_swa = [True] * n_layer  # window everywhere (unknown arch, no pattern key): nothing grows past the window
    else:
        is_swa = [False] * n_layer

    # per-layer KV heads (0 = recurrent/no-KV layer)
    hkv = list(n_head_kv_raw) if isinstance(n_head_kv_raw, list) else [n_head_kv] * n_layer
    hkv += [n_head_kv] * (n_layer - len(hkv))

    # hybrid recurrent/attention models: only every Nth layer has a KV cache
    interval = a("full_attention_interval")
    if not isinstance(n_head_kv_raw, list) and interval:
        keep = [((i + 1) % int(interval)) == 0 for i in range(n_layer)]
        hkv = [h if k else 0 for h, k in zip(hkv, keep)]

    kv_elems_growing = sum(h * (key_len + val_len) for i, h in enumerate(hkv[:n_layer]) if h and not is_swa[i])
    kv_elems_swa = sum(h * (key_swa + val_swa) for i, h in enumerate(hkv[:n_layer]) if h and is_swa[i])
    n_swa_layers = sum(1 for i, h in enumerate(hkv[:n_layer]) if h and is_swa[i])
    n_attn = sum(1 for h in hkv[:n_layer] if h) - n_swa_layers

    vocab = kv.get("tokenizer.ggml.tokens")
    n_vocab = vocab["__array_len__"] if isinstance(vocab, dict) else (len(vocab) if vocab else int(a("vocab_size", 0) or 0))

    ft = kv.get("general.file_type")
    quant = quant_from_filename(filename) or (FTYPE.get(ft) if ft is not None else None)

    params = kv.get("general.parameter_count")
    params_b = (params / 1e9) if isinstance(params, (int, float)) and params else None
    if params_b is None:
        m2 = re.search(r"(\d+(?:\.\d+)?)B", filename)
        if m2:
            params_b = float(m2.group(1))

    has_mtp = bool(a("nextn_predict_layers", 0)) or any("nextn" in k or ".mtp." in k for k in kv)

    return ModelShape(
        arch=arch,
        name=kv.get("general.name", filename or "?"),
        n_ctx_train=int(a("context_length", 0)),
        n_layer=n_layer,
        n_attn_layers=n_attn,
        n_head_kv=n_head_kv,
        key_len=key_len,
        val_len=val_len,
        n_embd=n_embd,
        n_vocab=int(n_vocab),
        file_type=ft,
        quant=quant,
        params_b=params_b,
        has_mtp=has_mtp,
        kv_elems_growing=kv_elems_growing,
        kv_elems_swa=kv_elems_swa,
        swa_window=swa_window,
        n_swa_layers=n_swa_layers,
        extra={
            "n_head": n_head,
            "full_attention_interval": interval,
            "expert_count": a("expert_count"),
            "expert_used_count": a("expert_used_count"),
            "rope_freq_base": a("rope.freq_base"),
        },
    )
