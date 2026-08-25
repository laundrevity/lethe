"""Streaming chat against llama-server with per-turn throughput + context stats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

import httpx


@dataclass
class TurnStats:
    prompt_n: int = 0
    cache_n: int = 0
    prompt_ms: float = 0.0
    predicted_n: int = 0
    predicted_ms: float = 0.0
    draft_n: int = 0
    draft_n_accepted: int = 0
    n_ctx: int = 0
    finish_reason: str | None = None

    @property
    def prompt_tps(self) -> float:
        return self.prompt_n / (self.prompt_ms / 1000) if self.prompt_ms else 0.0

    @property
    def gen_tps(self) -> float:
        return self.predicted_n / (self.predicted_ms / 1000) if self.predicted_ms else 0.0

    @property
    def ctx_used(self) -> int:
        return self.cache_n + self.prompt_n + self.predicted_n

    @property
    def accept_rate(self) -> float | None:
        return self.draft_n_accepted / self.draft_n if self.draft_n else None

    @classmethod
    def from_timings(cls, t: dict, n_ctx: int, finish: str | None) -> "TurnStats":
        return cls(
            prompt_n=t.get("prompt_n", 0),
            cache_n=t.get("cache_n", 0),
            prompt_ms=t.get("prompt_ms", 0.0),
            predicted_n=t.get("predicted_n", 0),
            predicted_ms=t.get("predicted_ms", 0.0),
            draft_n=t.get("draft_n", 0),
            draft_n_accepted=t.get("draft_n_accepted", 0),
            n_ctx=n_ctx,
            finish_reason=finish,
        )

    def line(self) -> str:
        parts = [
            f"prompt {self.prompt_n} tok @ {self.prompt_tps:,.0f} tok/s",
            f"gen {self.predicted_n} tok @ {self.gen_tps:.1f} tok/s",
        ]
        if self.draft_n:
            parts.append(f"mtp accept {self.accept_rate:.0%} ({self.draft_n_accepted}/{self.draft_n})")
        if self.n_ctx:
            parts.append(f"ctx {self.ctx_used:,}/{self.n_ctx:,} ({self.ctx_used / self.n_ctx:.1%})")
        if self.cache_n:
            parts.append(f"cache hit {self.cache_n}")
        if self.finish_reason and self.finish_reason != "stop":
            parts.append(f"finish={self.finish_reason}")
        return " · ".join(parts)


@dataclass
class Delta:
    content: str = ""
    reasoning: str = ""


def stream_chat(
    url: str,
    messages: list[dict],
    *,
    n_ctx: int,
    sampling: dict | None = None,
    max_tokens: int | None = None,
    chat_template_kwargs: dict | None = None,
) -> Iterator[Delta | TurnStats]:
    """Yield Delta chunks, then a final TurnStats."""
    body: dict = {"messages": messages, "stream": True, "timings_per_token": False}
    if sampling:
        body.update(sampling)
    if max_tokens:
        body["max_tokens"] = max_tokens
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs

    finish = None
    timings: dict = {}
    with httpx.Client(timeout=httpx.Timeout(600, connect=10)) as client:
        with client.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                r.read()
                raise RuntimeError(f"server {r.status_code}: {r.text[:500]}")
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                obj = json.loads(data)
                if "timings" in obj:
                    timings = obj["timings"]
                for ch in obj.get("choices", []):
                    d = ch.get("delta", {})
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    if d.get("content") or d.get("reasoning_content"):
                        yield Delta(content=d.get("content") or "", reasoning=d.get("reasoning_content") or "")
    yield TurnStats.from_timings(timings, n_ctx, finish)


THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repeat_penalty": 1.0}
INSTRUCT_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repeat_penalty": 1.0}
