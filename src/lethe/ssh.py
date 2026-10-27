"""Approved SSH execution without nested shell quoting or temporary scripts.

Only connection metadata and a fixed interpreter bootstrap appear in argv. The
source travels over stdin, and output stays in the existing bounded in-memory
runner. This controls the local SSH process, not the lifetime of remote jobs:
disconnecting on timeout or Ctrl-C cannot guarantee remote children are stopped.
User SSH configuration and the remote machine can have their own side effects.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from . import shell


# Keep aliases, identities and proxy configuration, but do not create connection
# sockets, background clients, forwarding listeners, or known_hosts updates.
# CheckHostIP can write known_hosts even with StrictHostKeyChecking=yes.
_SSH_OPTIONS = (
    "BatchMode=yes", "StrictHostKeyChecking=yes", "CheckHostIP=no", "UpdateHostKeys=no",
    "ControlMaster=no", "ControlPersist=no", "ClearAllForwardings=yes",
    "ForwardAgent=no", "ForwardX11=no", "Tunnel=no", "PermitLocalCommand=no",
    "AddKeysToAgent=no", "ForkAfterAuthentication=no", "SessionType=default",
    "StdinNull=no", "RemoteCommand=none", "LogLevel=ERROR", "LogVerbose=none",
)


def _validate_host(host: str) -> None:
    # Besides option injection, disallow shell syntax: configured ProxyCommand
    # strings may interpolate %h/%r without quoting them themselves.
    if not isinstance(host, str) or not host or len(host) > 512:
        raise ValueError("host must be an SSH alias, hostname or IP address, optionally user@host")
    parts = host.split("@")
    if len(parts) > 2 or (len(parts) == 2 and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*", parts[0])):
        raise ValueError("host must be an SSH alias, hostname or IP address, optionally user@host")
    address = parts[-1]
    if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", address):
        return
    if address.startswith("[") and address.endswith("]"):
        address = address[1:-1]
    if not re.fullmatch(r"[A-Fa-f0-9:]+(?:%[A-Za-z0-9_.-]+)?", address):
        raise ValueError("host contains invalid characters; use SSH config for ports and options")
    try:
        ipaddress.IPv6Address(address)
    except ValueError:
        raise ValueError("host must contain a valid IPv6 address") from None


def _quote_directory(cwd: str) -> str:
    if not isinstance(cwd, str) or not cwd or "\0" in cwd:
        raise ValueError("cwd must be a nonempty remote directory without NUL bytes")
    if cwd == "~":
        return '"$HOME"'
    if cwd.startswith("~/"):
        return '"$HOME"/' + shlex.quote(cwd[2:])
    if cwd.startswith("~"):
        raise ValueError("cwd supports ~ and ~/paths, but not ~other-user")
    # Avoid cd options, including cd '-' (OLDPWD), without relying on cd --.
    return shlex.quote("./" + cwd if cwd.startswith("-") else cwd)


def _prepare(
    host: str, bash: str | list[str] | None, python: str | None, cwd: str | None, timeout_seconds: int,
) -> tuple[str, str, list[str]]:
    _validate_host(host)
    if (bash is None) == (python is None):
        raise ValueError("provide exactly one of bash or python")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= shell.FULL_MAX_TIMEOUT:
        raise ValueError(f"timeout_seconds must be an integer from 1 to {shell.FULL_MAX_TIMEOUT}")
    directory = _quote_directory(cwd) if cwd is not None else None
    if python is not None:
        parts = [python]
        label = "python"
        interpreter = ["python3", "-B", "-u", "-"]
    elif isinstance(bash, list):
        parts = bash
        label = f"bash list ({len(parts)} commands, &&)"
        interpreter = ["bash", "--noprofile", "--norc", "-s"]
    else:
        parts = [bash]
        label = "bash script (-e)"
        interpreter = ["bash", "--noprofile", "--norc", "-e", "-s"]
    if not parts or any(not isinstance(part, str) or not part.strip() or "\0" in part for part in parts):
        raise ValueError("bash/python source must contain nonempty strings without NUL bytes")
    # Count user source, not the brace/newline scaffolding that Lethe adds.
    if sum(len(part.encode("utf-8")) for part in parts) > shell.MAX_COMMAND_BYTES:
        raise ValueError(f"bash/python source exceeds {shell.MAX_COMMAND_BYTES:,} UTF-8 bytes")
    if isinstance(bash, list):
        # Newlines protect closing braces from trailing comments and preserve
        # multiline commands/heredocs. Groups share variables and cwd. As with
        # a normal && list, each group's final status controls the next group.
        # Unlike raw scripts, do not use -e: compound entries must have the same
        # semantics whether they are first, middle, or last in the list.
        source = " &&\n".join("{\n" + part + "\n}" for part in parts) + "\n"
    else:
        source = parts[0]
    bootstrap = (f"CDPATH= cd {directory} && " if directory is not None else "") + "exec " + shlex.join(interpreter)
    remote_command = "exec " + shlex.join([
        "env", "HISTFILE=/dev/null", "BASH_ENV=/dev/null", "ENV=/dev/null", "PYTHONDONTWRITEBYTECODE=1",
        "/bin/sh", "-c", bootstrap,
    ])
    argv = ["ssh", "-T", "-S", "none"]
    for option in _SSH_OPTIONS:
        argv.extend(["-o", option])
    argv.extend(["--", host, remote_command])
    return label, source, argv


@dataclass(frozen=True)
class SSHRunner:
    """SSH shares full-shell approval/access, with source passed only on stdin."""

    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", shell._resolve_directory(self.root))

    def run(
        self, host: str, bash: str | list[str] | None = None, python: str | None = None,
        cwd: str | None = None, timeout_seconds: int = shell.FULL_DEFAULT_TIMEOUT,
    ) -> str:
        return json.dumps(self.execute(
            host=host, bash=bash, python=python, cwd=cwd, timeout_seconds=timeout_seconds,
        ), ensure_ascii=False)

    def execute(
        self, host: str, bash: str | list[str] | None = None, python: str | None = None,
        cwd: str | None = None, timeout_seconds: int = shell.FULL_DEFAULT_TIMEOUT,
    ) -> dict:
        _, source, argv = _prepare(host, bash, python, cwd, timeout_seconds)
        return shell._run_bounded(
            argv, source.encode("utf-8"), self.root, {**os.environ, "HISTFILE": "/dev/null"}, timeout_seconds,
            stop_on_output_limit=False,
        )

    def preview(
        self, host: str, bash: str | list[str] | None = None, python: str | None = None,
        cwd: str | None = None, timeout_seconds: int = shell.FULL_DEFAULT_TIMEOUT,
    ) -> str:
        label, source, _ = _prepare(host, bash, python, cwd, timeout_seconds)
        directory = ascii(cwd) if cwd is not None else "SSH login directory"
        return f"host={ascii(host)} · {label} · cwd={directory} · timeout={timeout_seconds}s\n{source}"
