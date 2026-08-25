"""Launch and talk to llama-server.

Privacy posture (the default, see README "Logless by default"):
  * llama-server runs with --log-disable, --no-webui, --no-slots; stdout/stderr go to /dev/null.
    Nothing it sees is written anywhere. `serve --log` opts in to a log file for debugging.
  * No pid file / state file. A running server is found by inspecting the process table.
  * Bound to 127.0.0.1 only.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .paths import llama_bin, state_dir

DEFAULT_PORT = 8080


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
    port: int = DEFAULT_PORT
    no_mmap: bool = False
    log: bool = False  # opt-in: keep llama-server's log in .lethe/server.log
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
        if not self.log:
            a.append("--log-disable")
        if self.no_mmap:
            a.append("--no-mmap")
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
    return state_dir() / "server.log"


def base_url(port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> str:
    return f"http://{host}:{port}"


def is_healthy(url: str, timeout: float = 1.0) -> bool:
    try:
        r = httpx.get(f"{url}/health", timeout=timeout)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def props(url: str) -> dict:
    return httpx.get(f"{url}/props", timeout=10).json()


def _argv_value(argv: list[str], flag: str) -> str | None:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def running_servers() -> list[dict]:
    """Find our llama-server processes from the process table (no state file)."""
    exe = str(llama_bin("llama-server"))
    try:
        out = subprocess.run(["ps", "-axo", "pid=,rss=,args="], capture_output=True, text=True, check=True).stdout
    except Exception:
        return []
    found = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid_s, rss_kb, args = parts
        if not args.startswith(exe):
            continue
        argv = shlex.split(args)
        port = int(_argv_value(argv, "--port") or DEFAULT_PORT)
        host = _argv_value(argv, "--host") or "127.0.0.1"
        found.append({
            "pid": int(pid_s),
            "rss": int(rss_kb) * 1024,
            "url": base_url(port, host),
            "port": port,
            "model": _argv_value(argv, "--model") or "?",
            "api_key": _argv_value(argv, "--api-key"),
            "argv": argv,
            "logging": "--log-disable" not in argv,
        })
    return found


def running_server(port: int | None = None) -> dict | None:
    """The healthy server on `port` (or any port if None)."""
    for rec in running_servers():
        if port is not None and rec["port"] != port:
            continue
        if is_healthy(rec["url"], timeout=3):
            return rec
    return None


def start(cfg: ServeConfig, wait: bool = True, on_wait=None) -> dict:
    """Spawn llama-server detached and wait for /health. Writes nothing unless cfg.log."""
    argv = cfg.argv()
    if cfg.log:
        sink = open(log_path(), "ab")
        sink.write(f"\n=== {time.strftime('%F %T')} ===\n$ {shlex.join(argv)}\n".encode())
        sink.flush()
    else:
        sink = subprocess.DEVNULL
    proc = subprocess.Popen(argv, stdout=sink, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    url = base_url(cfg.port, cfg.host)
    rec = {"pid": proc.pid, "url": url, "port": cfg.port, "model": str(cfg.model), "argv": argv, "logging": cfg.log}
    if not wait:
        return rec
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            hint = f"Log tail:\n{log_path().read_bytes()[-4000:].decode(errors='replace')}" if cfg.log else \
                "Re-run with --log (or --fg) to see llama-server's output."
            raise RuntimeError(f"llama-server exited with {proc.returncode}. {hint}")
        if is_healthy(url):
            return rec
        if on_wait:
            on_wait(time.time() - t0)
        time.sleep(0.5)


def stop(port: int | None = None) -> int:
    """SIGTERM every lethe llama-server (or just the one on `port`). Returns count stopped."""
    n = 0
    for rec in running_servers():
        if port is not None and rec["port"] != port:
            continue
        pid = rec["pid"]
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            continue
        for _ in range(40):
            try:
                os.kill(pid, 0)
                time.sleep(0.25)
            except ProcessLookupError:
                break
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        n += 1
    return n
