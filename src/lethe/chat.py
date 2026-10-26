"""Streaming chat against llama-server with per-turn throughput + context stats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

import httpx

from .server import auth_headers


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
    def ctx_frac(self) -> float:
        return self.ctx_used / self.n_ctx if self.n_ctx else 0.0

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

    def line(self, markup: bool = True) -> str:
        parts = [
            f"prompt {self.prompt_n} tok @ {self.prompt_tps:,.0f} tok/s",
            f"gen {self.predicted_n} tok @ {self.gen_tps:.1f} tok/s",
        ]
        if self.draft_n:
            parts.append(f"mtp accept {self.accept_rate:.0%} ({self.draft_n_accepted}/{self.draft_n})")
        if self.n_ctx:
            f = self.ctx_frac
            color = "green" if f < 0.6 else "yellow" if f < 0.85 else "red"
            ctx = f"ctx {self.ctx_used:,}/{self.n_ctx:,} ({f:.1%})"
            parts.append(f"[{color}]{ctx}[/]" if markup else ctx)
        if self.cache_n:
            parts.append(f"cache hit {self.cache_n}")
        if self.finish_reason and self.finish_reason != "stop":
            parts.append(f"finish={self.finish_reason}")
        return " · ".join(parts)


@dataclass
class Delta:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] | None = None  # complete accumulated calls, yielded once at stream end
    timings: dict | None = None  # server timings so far (cache_n, prompt_n, predicted_n, …), with every chunk
    progress: dict | None = None  # prompt processing so far: total, cache, processed (tokens), time_ms
    tool_call_deltas: list[dict] | None = None  # incremental fragments for display only, never execution


def stream_chat(
    url: str,
    messages: list[dict],
    *,
    n_ctx: int,
    sampling: dict | None = None,
    max_tokens: int | None = None,
    chat_template_kwargs: dict | None = None,
    api_key: str | None = None,
    tools: list[dict] | None = None,
) -> Iterator[Delta | TurnStats]:
    """Yield Delta chunks (including tool_call_deltas), then complete tool_calls and TurnStats.

    Every chunk asks for the server's running timings and prompt-processing progress, so the caller
    can show context use and speed while the reply is still being produced. Only the final tool_calls
    are executable: fragments may be incomplete, and a truncated stream must never run a tool.
    """
    body: dict = {"messages": messages, "stream": True, "timings_per_token": True, "return_progress": True}
    if sampling:
        body.update(sampling)
    if max_tokens:
        body["max_tokens"] = max_tokens
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs
    if tools:
        body["tools"] = tools

    finish = None
    timings: dict = {}
    calls: dict[int, dict] = {}
    done = False
    with httpx.Client(timeout=httpx.Timeout(3600, connect=10), headers=auth_headers(api_key)) as client:
        with client.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                r.read()
                raise RuntimeError(f"server {r.status_code}: {r.text[:500]}")
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    done = True
                    break
                obj = json.loads(data)
                chunk_timings = obj.get("timings") or None
                progress = obj.get("prompt_progress") or None
                if chunk_timings:
                    timings = chunk_timings
                emitted = False
                for ch in obj.get("choices", []):
                    d = ch.get("delta", {})
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    for tc in d.get("tool_calls") or []:
                        i = tc.get("index", 0)
                        slot = calls.setdefault(i, {"id": f"call_{i}", "type": "function", "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        f = tc.get("function") or {}
                        slot["function"]["name"] += f.get("name") or ""
                        slot["function"]["arguments"] += f.get("arguments") or ""
                    if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                        yield Delta(content=d.get("content") or "", reasoning=d.get("reasoning_content") or "",
                                    tool_call_deltas=d.get("tool_calls") or None,
                                    timings=chunk_timings, progress=progress)
                        emitted = True
                if (chunk_timings or progress) and not emitted:
                    yield Delta(timings=chunk_timings, progress=progress)
    if calls:
        if not done or finish not in {"tool_calls", "stop"}:
            raise RuntimeError("incomplete tool-call response; no tools were executed")
        yield Delta(tool_calls=[calls[i] for i in sorted(calls)])
    yield TurnStats.from_timings(timings, n_ctx, finish)


def complete(
    url: str,
    messages: list[dict],
    *,
    sampling: dict | None = None,
    max_tokens: int | None = None,
    chat_template_kwargs: dict | None = None,
    api_key: str | None = None,
) -> tuple[str, dict]:
    """Non-streaming completion. Returns (content, timings)."""
    body: dict = {"messages": messages, "stream": False}
    if sampling:
        body.update(sampling)
    if max_tokens:
        body["max_tokens"] = max_tokens
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs
    r = httpx.post(f"{url}/v1/chat/completions", json=body, headers=auth_headers(api_key), timeout=3600)
    if r.status_code != 200:
        raise RuntimeError(f"server {r.status_code}: {r.text[:500]}")
    obj = r.json()
    return obj["choices"][0]["message"].get("content") or "", obj.get("timings", {})


def tokenize(url: str, text: str, api_key: str | None = None) -> int:
    r = httpx.post(f"{url}/tokenize", json={"content": text}, headers=auth_headers(api_key), timeout=600)
    r.raise_for_status()
    return len(r.json().get("tokens", []))
