"""Per-model-family serving profiles: the sampler the model card recommends, and how thinking is toggled.

Chosen from the GGUF architecture / name, overridable per flag on `lethe chat`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ggufmeta import ModelShape


@dataclass(frozen=True)
class Profile:
    name: str
    thinking: dict  # sampler when reasoning is on
    instruct: dict  # sampler when reasoning is off
    supports_thinking: bool = True
    disable_thinking_kwargs: dict = field(default_factory=lambda: {"enable_thinking": False})
    notes: str = ""

    def sampling(self, thinking: bool) -> dict:
        return dict(self.thinking if (thinking and self.supports_thinking) else self.instruct)


_QWEN3 = Profile(
    "qwen3",
    thinking={"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repeat_penalty": 1.0},
    instruct={"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repeat_penalty": 1.0},
    notes="Qwen3.x model card: thinking temp 1.0/top_p 0.95/top_k 20; non-thinking temp 0.7/top_p 0.8/presence 1.5",
)
_GEMMA = Profile(
    "gemma",
    thinking={"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0, "repeat_penalty": 1.0},
    instruct={"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0, "repeat_penalty": 1.0},
    notes="Gemma model card: temp 1.0/top_p 0.95/top_k 64",
)
_MINIMAX = Profile(
    "minimax-m2",
    thinking={"temperature": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.0, "repeat_penalty": 1.0},
    instruct={"temperature": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.0, "repeat_penalty": 1.0},
    notes="MiniMax-M2 model card: temp 1.0/top_p 0.95/top_k 40; the model always thinks",
)
_DEEPSEEK = Profile(
    "deepseek",
    thinking={"temperature": 0.6, "top_p": 0.95, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0},
    instruct={"temperature": 0.6, "top_p": 0.95, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0},
    notes="DeepSeek R1/V3 guidance: temp 0.6/top_p 0.95",
)
_GLM = Profile(
    "glm",
    thinking={"temperature": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.0, "repeat_penalty": 1.0},
    instruct={"temperature": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.0, "repeat_penalty": 1.0},
    notes="GLM-4.5 model card: temp 1.0/top_p 0.95",
)
_DEFAULT = Profile(
    "default",
    thinking={"temperature": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.0},
    instruct={"temperature": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.0},
    notes="generic conservative defaults; no model-card guidance known for this architecture",
)

_BY_ARCH = {
    "qwen3": _QWEN3, "qwen3moe": _QWEN3, "qwen3next": _QWEN3, "qwen35": _QWEN3, "qwen35moe": _QWEN3, "qwen3vl": _QWEN3, "qwen3vlmoe": _QWEN3,
    "gemma3": _GEMMA, "gemma3n": _GEMMA, "gemma4": _GEMMA,
    "minimax-m2": _MINIMAX, "minimaxm2": _MINIMAX,
    "deepseek2": _DEEPSEEK, "deepseek3": _DEEPSEEK,
    "glm4": _GLM, "glm4moe": _GLM,
}


def profile_for(shape: ModelShape) -> Profile:
    p = _BY_ARCH.get(shape.arch.lower())
    if p:
        return p
    name = (shape.name or "").lower()
    for key, prof in (("qwen", _QWEN3), ("gemma", _GEMMA), ("minimax", _MINIMAX), ("deepseek", _DEEPSEEK), ("glm", _GLM)):
        if key in name:
            return prof
    return _DEFAULT


def describe(s: dict) -> str:
    keys = ["temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty"]
    short = {"temperature": "temp", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p", "presence_penalty": "presence", "repeat_penalty": "repeat"}
    return " ".join(f"{short[k]} {s[k]}" for k in keys if k in s and not (k == "repeat_penalty" and s[k] == 1.0) and not (k == "presence_penalty" and s[k] == 0.0))
