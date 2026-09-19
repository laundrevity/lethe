"""Opt-in, read-only command execution under macOS Seatbelt.

There is deliberately no unsandboxed fallback or writable mode. Commands, the
profile, and output stay in memory; no shell history or temporary scripts.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
MAX_COMMAND_BYTES = 8_000
MAX_OUTPUT_BYTES = 16_000
MAX_TIMEOUT = 30

# Paths are passed as sandbox parameters, never interpolated into Scheme code.
# No network, Mach services (including clipboard/Apple Events), arbitrary device
# access, or file writes are granted. Descendants inherit these restrictions.
PROFILE = r'''(version 1)
(deny default (with no-log))
(allow process-exec process-fork)
(allow signal (target same-sandbox))
(allow process-info* (target same-sandbox))
(allow system-mac-syscall (mac-policy-name "vnguard"))
(allow system-mac-syscall
  (require-all (mac-policy-name "Sandbox") (mac-syscall-number 67)))
(allow sysctl-read
  (sysctl-name-prefix "hw.")
  (sysctl-name "kern.osrelease")
  (sysctl-name "kern.osversion")
  (sysctl-name "kern.ostype")
  (sysctl-name "kern.version")
  (sysctl-name "kern.argmax")
  (sysctl-name "kern.usrstack64"))
(allow file-read* file-map-executable
  (subpath (param "WORKSPACE"))
  (subpath "/bin")
  (subpath "/usr/bin")
  (subpath "/usr/lib")
  (subpath "/usr/share")
  (subpath "/System/Library")
  (subpath "/System/Volumes/Preboot/Cryptexes/OS")
  (subpath "/Library/Apple/System/Library"))
(allow file-read-metadata
  (path-ancestors (param "WORKSPACE"))
  (literal "/tmp") (literal "/var") (literal "/etc"))
(allow file-read* (literal "/") (literal "/private/var/select/sh"))
(allow file-read* (literal "/dev/null") (literal "/dev/zero")
  (literal "/dev/random") (literal "/dev/urandom"))
(allow file-write-data (literal "/dev/null"))
'''


class SandboxUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ShellSandbox:
    root: Path

    def __post_init__(self):
        root = self.root.expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("shell directory must be an existing directory")
        # Avoid accidentally exposing an entire machine or home directory.
        if root == Path.home().resolve() or root in Path.home().resolve().parents:
            raise ValueError("choose a project directory, not your home directory or its parents")
        object.__setattr__(self, "root", root)

    def check_available(self) -> None:
        """Probe the real sandbox before starting/reusing a model server."""
        result = self.execute("printf lethe-sandbox-ok", timeout_seconds=5)
        if result["exit_code"] != 0 or result["stdout"] != "lethe-sandbox-ok":
            detail = result["stderr"].strip() or f"exit {result['exit_code']}"
            raise SandboxUnavailable(f"macOS sandbox could not start ({detail}); shell tool disabled")

    def run(self, command: str, cwd: str = ".", timeout_seconds: int = 10) -> str:
        """JSON tool result; errors are handled by the REPL like other tool errors."""
        return json.dumps(self.execute(command, cwd, timeout_seconds), ensure_ascii=False)

    def execute(self, command: str, cwd: str = ".", timeout_seconds: int = 10) -> dict:
        if sys.platform != "darwin" or not SANDBOX_EXEC.is_file():
            raise SandboxUnavailable("shell tools require macOS /usr/bin/sandbox-exec; no unsandboxed fallback")
        if not isinstance(command, str) or not command.strip() or "\0" in command:
            raise ValueError("command must be a nonempty string without NUL bytes")
        if len(command.encode("utf-8")) > MAX_COMMAND_BYTES:
            raise ValueError(f"command exceeds {MAX_COMMAND_BYTES:,} bytes")
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= MAX_TIMEOUT:
            raise ValueError(f"timeout_seconds must be an integer from 1 to {MAX_TIMEOUT}")
        if not isinstance(cwd, str) or "\0" in cwd:
            raise ValueError("cwd must be a directory within the shell workspace")
        workdir = (self.root / cwd).resolve(strict=True)
        if not workdir.is_relative_to(self.root) or not workdir.is_dir():
            raise ValueError("cwd must be a directory within the shell workspace")
        # The policy remains pinned to the original canonical root if a directory
        # is renamed or replaced with a symlink while the conversation is running.
        if self.root.resolve(strict=True) != self.root:
            raise ValueError("shell workspace changed; restart with the intended directory")
        env = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C", "LC_ALL": "C", "TZ": "UTC",
            "HISTFILE": "/dev/null", "TERM": "dumb",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0",
        }
        # The trusted preamble lowers hard limits before any model text runs.
        # No inherited tokens, proxies, SSH agents, startup files, or Python paths.
        script = (
            "ulimit -S -H -c 0 || exit 125\n"
            f"ulimit -S -H -t {timeout_seconds} || exit 125\n"
            "ulimit -S -H -n 128 || exit 125\n"
            "ulimit -S -H -f 0 || exit 125\n"
            + command + "\n"
        ).encode("utf-8")
        argv = [str(SANDBOX_EXEC), "-D", f"WORKSPACE={self.root}", "-p", PROFILE, "/bin/sh", "-s"]
        return _run_bounded(argv, script, workdir, env, timeout_seconds)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run_bounded(argv: list[str], script: bytes, cwd: Path, env: dict, timeout: int) -> dict:
    """Drain both pipes without unbounded communicate() buffers, even on a flood.

    A new process group lets timeout, output overflow, and Ctrl-C also stop normal
    child processes. Detached processes are unsupported (see README).
    """
    started = time.monotonic()
    output = {"stdout": bytearray(), "stderr": bytearray()}
    total = 0
    timed_out = truncated = False
    with subprocess.Popen(
        argv, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
    ) as proc:
        try:
            with selectors.DefaultSelector() as selector:
                for stream, name in ((proc.stdin, "stdin"), (proc.stdout, "stdout"), (proc.stderr, "stderr")):
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_WRITE if name == "stdin" else selectors.EVENT_READ, name)
                pending = memoryview(script)
                while selector.get_map():
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        timed_out = True
                        break
                    for key, _ in selector.select(min(remaining, 0.1)):
                        if key.data == "stdin":
                            try:
                                pending = pending[os.write(key.fd, pending):]
                            except BrokenPipeError:
                                pending = pending[:0]
                            if not pending:
                                selector.unregister(key.fileobj)
                                key.fileobj.close()
                            continue
                        chunk = os.read(key.fd, 4096)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        room = MAX_OUTPUT_BYTES - total
                        output[key.data].extend(chunk[:room])
                        total += min(room, len(chunk))
                        if len(chunk) > room:
                            truncated = True
                            break
                    if truncated:
                        break
                if not timed_out and not truncated:
                    try:
                        proc.wait(timeout=max(0.001, timeout - (time.monotonic() - started)))
                    except subprocess.TimeoutExpired:
                        timed_out = True
        finally:
            # Also clean up background jobs on a successful shell exit, and on
            # KeyboardInterrupt. No subprocess is deliberately left running.
            _kill_group(proc)
            proc.wait()
    return {
        "exit_code": proc.returncode,
        "stdout": output["stdout"].decode("utf-8", errors="replace"),
        "stderr": output["stderr"].decode("utf-8", errors="replace"),
        "timed_out": timed_out, "truncated": truncated,
    }
