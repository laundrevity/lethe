"""Run llama-server for one `lethe` session and talk to it.

The server belongs to the `lethe` process that started it: it comes up when a chat
(or benchmark) starts and is killed when that ends, however it ends. There is no
daemon to discover, no pid file, and nothing left holding the model afterwards.

Privacy posture (the default, see README "Logless by default"):
  * llama-server runs with --log-disable, --no-webui, --no-slots; its output goes to
    /dev/null unless `--debug-server` shows it on the terminal. Nothing is written anywhere.
  * Bound to 127.0.0.1 on a free port, with a per-launch random API key.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import httpx

from .paths import llama_bin, state_dir


@dataclass
class ServeConfig:
    model: Path
    n_ctx: int  # 0 = model's native maximum
    draft: Path | None = None  # FastMTP-style sidecar
    mtp: bool = True  # --spec-type draft-mtp (embedded head, or `draft` if given)
    draft_n_max: int = 3
    mmproj: Path | None = None
    kv_type: str = "f16"  # f16 | q8_0 | q4_0
    flash_attn: bool = True
    n_gpu_layers: str = "all"
    batch: int = 2048
    ubatch: int = 512
    parallel: int = 1
    thinking: bool = True
    reasoning_effort: str | None = None  # xhigh|medium|low (Qwen3.8)
    host: str = "127.0.0.1"
    port: int = 0  # 0: pick a free one when the session starts
    no_mmap: bool = False
    debug: bool = False  # show llama-server's own output on the terminal (nothing is written to disk)
    api_key: str | None = None  # per-launch random key: blocks other local software / web pages from using the model
    extra: list[str] = field(default_factory=list)

    def argv(self) -> list[str]:
        a = [
            str(llama_bin("llama-server")),
            "--model", str(self.model),
            "--ctx-size", str(self.n_ctx),
            "--n-gpu-layers", self.n_gpu_layers,
            "--batch-size", str(self.batch),
            "--ubatch-size", str(self.ubatch),
            "--parallel", str(self.parallel),
            "--flash-attn", "on" if self.flash_attn else "off",
            "--cache-type-k", self.kv_type,
            "--cache-type-v", self.kv_type,
            "--jinja",
            "--host", self.host,
            "--port", str(self.port),
            # --- traceless defaults ---
            "--no-webui",   # the browser UI keeps chat history in the browser; we never serve it
            "--no-slots",   # /slots would expose the live prompt text over HTTP
        ]
        if not self.debug:
            a.append("--log-disable")
        if self.api_key:
            a += ["--api-key", self.api_key]
        if self.no_mmap:
            a += ["--load-mode", "none"]
        if self.mtp:
            a += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(self.draft_n_max), "--spec-draft-p-min", "0"]
            if self.draft:
                a += ["--spec-draft-model", str(self.draft), "--spec-draft-ngl", "all"]
        if self.mmproj:
            a += ["--mmproj", str(self.mmproj)]
        if self.thinking:
            a += ["--reasoning", "on", "--reasoning-format", "deepseek", "--reasoning-preserve"]
            if self.reasoning_effort:
                a += ["--reasoning-effort", self.reasoning_effort]
        else:
            a += ["--chat-template-kwargs", json.dumps({"enable_thinking": False})]
        a += self.extra
        return a


def log_path() -> Path:
    """Where older versions kept an opt-in server log; only `traces` still looks for it."""
    return state_dir() / "server.log"


def base_url(port: int, host: str = "127.0.0.1") -> str:
    return f"http://{host}:{port}"


def is_healthy(url: str, timeout: float = 1.0) -> bool:
    try:
        r = httpx.get(f"{url}/health", timeout=timeout)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def props(url: str, api_key: str | None = None) -> dict:
    return httpx.get(f"{url}/props", headers=auth_headers(api_key), timeout=10).json()


def new_api_key() -> str:
    import secrets

    return secrets.token_urlsafe(24)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Session:
    """A llama-server that lives exactly as long as the `with` block that starts it.

    Leaving the block kills it, whether the block ends normally, with an exception,
    with Ctrl-C or Ctrl-D, or because the process got SIGTERM or SIGHUP (closed
    terminal). Only SIGKILL of lethe itself can leave the server behind.
    """

    def __init__(self, cfg: ServeConfig, on_wait=None):
        if not cfg.port:
            cfg = replace(cfg, port=free_port())
        self.cfg = cfg
        self.url = base_url(cfg.port, cfg.host)
        self.api_key = cfg.api_key
        self.model = str(cfg.model)
        self.on_wait = on_wait
        self.proc: subprocess.Popen | None = None
        self._previous_handlers: dict[int, object] = {}

    def __enter__(self) -> "Session":
        argv = self.cfg.argv()
        # Its own process group: the terminal's Ctrl-C reaches lethe, which decides what to do;
        # it never half-kills the server.
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=None if self.cfg.debug else subprocess.DEVNULL, start_new_session=True,
        )
        self._arm()
        t0 = time.time()
        try:
            while not is_healthy(self.url):
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited with {self.proc.returncode}; run with --debug-server to see its output"
                    )
                if self.on_wait:
                    self.on_wait(time.time() - t0)
                time.sleep(0.5)
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _arm(self) -> None:
        """Make sure the server dies with us, not only when the `with` block is left cleanly."""
        atexit.register(self.stop)
        for signum in (signal.SIGTERM, signal.SIGHUP):
            try:
                self._previous_handlers[signum] = signal.signal(signum, self._on_signal)
            except ValueError:  # not the main thread
                pass

    def _disarm(self) -> None:
        atexit.unregister(self.stop)
        for signum, previous in self._previous_handlers.items():
            signal.signal(signum, previous)
        self._previous_handlers.clear()

    def _on_signal(self, signum, frame) -> None:
        self.stop()
        raise SystemExit(128 + signum)

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        self._disarm()
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            except ProcessLookupError:
                pass
