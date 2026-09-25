import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lethe import shell


@pytest.fixture
def sandbox(tmp_path, request):
    if not request.config.getoption("--run-sandbox"):
        pytest.skip("real macOS sandbox tests: pass --run-sandbox")
    if sys.platform != "darwin":
        pytest.fail("--run-sandbox requires macOS")
    root = tmp_path / 'project with spaces " and )'
    root.mkdir()
    instance = shell.ShellSandbox(root)
    instance.check_available()  # fail, never skip, if the actual policy cannot start
    return instance


def test_reject_broad_or_missing_roots(tmp_path):
    for root in (Path("/"), Path.home(), Path.home().parent):
        with pytest.raises(ValueError, match="project directory"):
            shell.ShellSandbox(root)
    with pytest.raises(FileNotFoundError):
        shell.ShellSandbox(tmp_path / "missing")


def test_fail_closed_off_macos(tmp_path, monkeypatch):
    monkeypatch.setattr(shell.sys, "platform", "linux")
    monkeypatch.setattr(shell.subprocess, "Popen", lambda *a, **kw: pytest.fail("must never launch"))
    with pytest.raises(shell.SandboxUnavailable):
        shell.ShellSandbox(tmp_path).execute("true")


def test_failed_probe_disables_shell(tmp_path, monkeypatch):
    monkeypatch.setattr(shell.ShellSandbox, "execute", lambda *a, **kw: {
        "exit_code": 71, "stdout": "", "stderr": "sandbox_apply: Operation not permitted",
    })
    with pytest.raises(shell.SandboxUnavailable, match="shell tool disabled"):
        shell.ShellSandbox(tmp_path).check_available()


@pytest.mark.parametrize("kwargs", [
    {"command": ""}, {"command": "\0"}, {"command": 5}, {"command": "x" * 8001},
    {"command": "true", "timeout_seconds": 0}, {"command": "true", "timeout_seconds": 31},
    {"command": "true", "timeout_seconds": True}, {"command": "true", "timeout_seconds": 1.5},
    {"command": "true", "cwd": ".."}, {"command": "true", "cwd": "/"},
])
def test_invalid_arguments_never_execute(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(shell.sys, "platform", "darwin")
    monkeypatch.setattr(shell, "SANDBOX_EXEC", Path(__file__))
    monkeypatch.setattr(shell, "_run_bounded", lambda *a, **kw: pytest.fail("must never launch"))
    with pytest.raises(ValueError):
        shell.ShellSandbox(tmp_path).execute(**kwargs)


def test_environment_and_launch_are_fixed(tmp_path, monkeypatch):
    monkeypatch.setenv("LETHE_TEST_SECRET", "do-not-inherit")
    monkeypatch.setenv("BASH_ENV", "/malicious/startup")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")
    monkeypatch.setattr(shell.sys, "platform", "darwin")
    monkeypatch.setattr(shell, "SANDBOX_EXEC", Path(__file__))
    captured = []
    monkeypatch.setattr(shell, "_run_bounded", lambda *args: captured.append(args))
    shell.ShellSandbox(tmp_path).execute("printf 'private command'", timeout_seconds=3)
    argv, script, cwd, env, timeout = captured[0]
    assert argv[-2:] == ["/bin/sh", "-s"]
    assert all("private command" not in arg for arg in argv)
    assert script.endswith(b"printf 'private command'\n")
    assert cwd == tmp_path.resolve() and timeout == 3
    assert not {"LETHE_TEST_SECRET", "BASH_ENV", "HTTPS_PROXY", "SSH_AUTH_SOCK", "HOME"}.intersection(env)
    assert env["PATH"] == "/usr/bin:/bin"


def test_full_shell_writes_and_inherits_environment_outside_starting_directory(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    bindir = tmp_path / "tools"
    bindir.mkdir()
    helper = bindir / "lethe-test-helper"
    helper.write_text('#!/bin/sh\nprintf "%s" "$LETHE_TEST_VALUE"\n')
    helper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:/usr/bin:/bin")
    monkeypatch.setenv("LETHE_TEST_VALUE", "inherited-value")
    monkeypatch.setattr(shell, "SANDBOX_EXEC", tmp_path / "no-sandbox-exec")
    full = shell.FullShell(root)
    full.check_available()
    result = full.execute("lethe-test-helper > written; cat written", cwd="..")
    assert result["exit_code"] == 0 and result["stdout"] == "inherited-value"
    assert (tmp_path / "written").read_text() == "inherited-value"
    assert full.execute("pwd", cwd=str(tmp_path))["stdout"].strip() == str(tmp_path.resolve())
    assert full.execute("pwd", cwd="~")["stdout"].strip() == str(Path.home().resolve())
    assert shell.FullShell(Path.home()).root == Path.home().resolve()


def test_full_shell_uses_longer_timeouts_without_sandbox_limits(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(shell, "_run_bounded", lambda *args, **kw: captured.append((args, kw)))
    full = shell.FullShell(tmp_path)
    full.execute("printf test")
    full.execute("printf test", timeout_seconds=600)
    for (argv, script, cwd, env, timeout), kwargs in captured:
        assert argv == ["/bin/sh", "-s"] and script == b"printf test\n"
        assert env["HISTFILE"] == "/dev/null"
        assert kwargs == {"stop_on_output_limit": False}
    assert [args[-1] for args, kw in captured] == [120, 600]


@pytest.mark.parametrize("kwargs", [
    {"command": ""}, {"command": "\0"}, {"command": 5}, {"command": "x" * 8001},
    {"command": "true", "timeout_seconds": 0}, {"command": "true", "timeout_seconds": 601},
    {"command": "true", "timeout_seconds": True}, {"command": "true", "timeout_seconds": 1.5},
    {"command": "true", "cwd": "\0"}, {"command": "true", "cwd": 1},
])
def test_full_shell_validates_arguments_before_execution(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(shell, "_run_bounded", lambda *a, **kw: pytest.fail("must never launch"))
    with pytest.raises(ValueError):
        shell.FullShell(tmp_path).execute(**kwargs)


def test_full_shell_keeps_running_after_output_is_truncated(tmp_path):
    result = shell.FullShell(tmp_path).execute(
        "i=0; while [ $i -lt 4000 ]; do printf 'verbose build output\\n'; i=$((i+1)); done\n"
        "printf finished > result.txt"
    )
    assert result["exit_code"] == 0 and result["truncated"] and not result["timed_out"]
    assert len(result["stdout"].encode()) + len(result["stderr"].encode()) == shell.MAX_OUTPUT_BYTES
    assert (tmp_path / "result.txt").read_text() == "finished"
    result = shell.FullShell(tmp_path).execute("yes output", timeout_seconds=1)
    assert result["truncated"] and result["timed_out"]


def test_full_shell_allows_loopback(request, tmp_path):
    if not request.config.getoption("--run-sandbox"):
        pytest.skip("real shell networking test: pass --run-sandbox outside any outer sandbox")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        result = shell.FullShell(tmp_path).execute(f"/usr/bin/nc -z -w 2 127.0.0.1 {port}")
        assert result["exit_code"] == 0, result
        listener.settimeout(1)
        connection, address = listener.accept()
        connection.close()


def test_runner_caps_output_and_time(tmp_path):
    start = time.monotonic()
    result = shell._run_bounded(["/bin/sh"], b"while :; do printf 1234567890; done\n", tmp_path, {}, 2)
    assert result["truncated"] and not result["timed_out"]
    assert len(result["stdout"]) + len(result["stderr"]) <= shell.MAX_OUTPUT_BYTES
    result = shell._run_bounded(["/bin/sh"], b"sleep 20 & wait\n", tmp_path, {}, 1)
    assert result["timed_out"]
    assert time.monotonic() - start < 5


def test_runner_cleans_up_on_interrupt(tmp_path, monkeypatch):
    spawned = []
    popen = subprocess.Popen
    selector_type = shell.selectors.DefaultSelector

    def launch(*args, **kwargs):
        proc = popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    def interrupting_selector():
        selector = selector_type()
        def interrupt(*args):
            raise KeyboardInterrupt
        selector.select = interrupt
        return selector

    monkeypatch.setattr(shell.subprocess, "Popen", launch)
    monkeypatch.setattr(shell.selectors, "DefaultSelector", interrupting_selector)
    with pytest.raises(KeyboardInterrupt):
        shell._run_bounded(["/bin/sh"], b"sleep 20\n", tmp_path, {}, 10)
    assert spawned[0].poll() is not None


def test_native_reads_pipes_stderr_and_exit_status(sandbox):
    (sandbox.root / "input.txt").write_text("one\ntwo\nthree\n")
    result = sandbox.execute("cat input.txt | wc -l; printf error >&2; exit 7")
    assert result["exit_code"] == 7
    assert result["stdout"].strip() == "3"
    assert result["stderr"] == "error"
    assert not result["timed_out"] and not result["truncated"]
    subdir = sandbox.root / "subdir"
    subdir.mkdir()
    assert sandbox.execute("pwd", cwd="subdir")["stdout"].strip() == str(subdir)


def test_native_denies_reads_outside_root_and_symlink_escape(sandbox):
    outside = sandbox.root.parent / "secret"
    outside.write_text("outside-secret")
    (sandbox.root / "escape").symlink_to(outside)
    (sandbox.root / "outside-dir").symlink_to(outside.parent, target_is_directory=True)
    for path in (str(outside), "escape"):
        result = sandbox.execute(f"cat {shlex.quote(path)}")
        assert result["exit_code"] != 0 and "outside-secret" not in result["stdout"]
    with pytest.raises(ValueError, match="within the shell workspace"):
        sandbox.execute("pwd", cwd="outside-dir")


def test_native_denies_writes_and_child_escape(sandbox):
    original = sandbox.root / "original"
    original.write_text("keep")
    for command in (
        "printf changed > original", "touch created", "mkdir newdir", "rm original",
        "/bin/sh -c 'printf changed > original'", "ln -s /etc/passwd link",
    ):
        result = sandbox.execute(command)
        assert result["exit_code"] != 0, command
    assert original.read_text() == "keep"
    assert {p.name for p in sandbox.root.iterdir()} == {"original"}
    outside = sandbox.root.parent / "outside-new"
    assert sandbox.execute(f"touch {shlex.quote(str(outside))}")["exit_code"] != 0
    assert not outside.exists()


def test_native_denies_loopback_and_unix_sockets(sandbox):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        result = sandbox.execute(f"/usr/bin/curl -q --noproxy '*' --max-time 2 http://127.0.0.1:{port}/")
        assert result["exit_code"] != 0
        listener.settimeout(0.1)
        with pytest.raises(TimeoutError):
            listener.accept()
    # Use a short socket path (Darwin's sockaddr_un length is only 104 bytes).
    import tempfile
    with tempfile.TemporaryDirectory(prefix="lethe-sock-", dir="/tmp") as tmp:
        socket_path = str(Path(tmp).resolve() / "s")
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(socket_path)
            listener.listen()
            # The socket itself is readable in this workspace, so this checks
            # network enforcement rather than relying on a denied file path.
            socket_sandbox = shell.ShellSandbox(Path(tmp))
            result = socket_sandbox.execute(f"/usr/bin/nc -z -U {shlex.quote(socket_path)}")
            assert result["exit_code"] != 0
            listener.settimeout(0.1)
            with pytest.raises(TimeoutError):
                listener.accept()


def test_native_no_inherited_secrets_or_host_process_access(sandbox, monkeypatch):
    monkeypatch.setenv("LETHE_TEST_SECRET", "outside-secret")
    result = sandbox.execute("env")
    assert result["exit_code"] == 0 and "outside-secret" not in result["stdout"]
    # ps could otherwise expose llama-server's API key from its argv.
    result = sandbox.execute(f"ps -p {os.getpid()} -o command=")
    assert result["exit_code"] != 0 and not result["stdout"].strip()
    result = sandbox.execute(f"kill -0 {os.getpid()}")
    assert result["exit_code"] != 0


def test_native_timeout_and_output_limit(sandbox):
    result = sandbox.execute("printf started; sleep 20 & wait", timeout_seconds=1)
    assert result["timed_out"] and result["stdout"] == "started"
    result = sandbox.execute("yes lots-of-output")
    assert result["truncated"]
    assert len(result["stdout"].encode()) + len(result["stderr"].encode()) <= shell.MAX_OUTPUT_BYTES
