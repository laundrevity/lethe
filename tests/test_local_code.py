import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lethe import shell


@pytest.fixture
def runner(tmp_path, monkeypatch):
    # Use the test interpreter while still exercising inherited PATH lookup.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python3").symlink_to(sys.executable)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    root = tmp_path / "workspace"
    root.mkdir()
    return shell.FullShell(root)


def test_python_source_survives_quotes_newlines_and_unicode(runner):
    value = "quotes ' \"; backslash \\; café 👋\nsecond line\n"
    result = json.loads(runner.run_python(
        "import sys\n"
        f"sys.stdout.write({value!r})\n"
        "sys.stderr.write('error detail')\n"
        "sys.exit(7)"
    ))
    assert result == {"stdout": value, "stderr": "error detail", "exit_code": 7,
                      "timed_out": False, "truncated": False}
    assert list(runner.root.iterdir()) == []


def test_python_import_does_not_write_bytecode(runner):
    (runner.root / "helper.py").write_text("value = 'imported'\n")
    result = runner.execute_python("import helper; print(helper.value)")
    assert result["exit_code"] == 0 and result["stdout"] == "imported\n"
    assert {path.name for path in runner.root.iterdir()} == {"helper.py"}


def test_code_uses_inherited_environment_and_path_without_startup_files(runner, monkeypatch):
    monkeypatch.setenv("LETHE_CODE_TEST", "inherited")
    startup = runner.root / "startup.sh"
    startup.write_text("printf startup-ran\n")
    monkeypatch.setenv("BASH_ENV", str(startup))
    monkeypatch.setenv("ENV", str(startup))
    helper = runner.root.parent / "bin" / "lethe-code-helper"
    helper.write_text('#!/bin/sh\nprintf "%s" "$LETHE_CODE_TEST"\n')
    helper.chmod(0o755)
    for result in (
        runner.execute_python("import os; print(os.environ['LETHE_CODE_TEST'], end='')"),
        runner.execute_bash(["lethe-code-helper"]),
    ):
        assert result["exit_code"] == 0 and result["stdout"] == "inherited"


def test_python_and_bash_cwd_is_literal_and_can_leave_starting_directory(runner, monkeypatch):
    directory = runner.root.parent / "other ' $(touch no); directory"
    directory.mkdir()
    monkeypatch.setenv("HOME", str(directory))
    for cwd in (str(directory), "../" + directory.name, "~"):
        for result in (
            runner.execute_python("import os; print(os.getcwd())", cwd=cwd),
            runner.execute_bash(["pwd"], cwd=cwd),
        ):
            assert result["exit_code"] == 0 and result["stdout"] == str(directory) + "\n"
    assert not (runner.root / "no").exists()
    assert not (directory / "no").exists()


def test_bash_commands_share_variables_directory_and_multiline_constructs(runner):
    (runner.root / "subdir").mkdir()
    result = json.loads(runner.run_bash([
        "value='hello world' # trailing comment",
        "cd subdir",
        'printf "%s:%s\\n" "$value" "${PWD##*/}"',
        "for item in first second; do\n  printf '%s\\n' \"$item\"\ndone",
        "cat <<'END'\nquotes ' \" and $literal\nEND",
    ]))
    assert result["exit_code"] == 0
    assert result["stdout"] == "hello world:subdir\nfirst\nsecond\nquotes ' \" and $literal\n"


@pytest.mark.parametrize("commands,stdout,exit_code", [
    (["printf before", "false", "printf after"], "before", 1),
    (["false; printf recovered"], "recovered", 0),
    (["true", "false; printf recovered"], "recovered", 0),
    (["false; printf recovered", "printf after"], "recoveredafter", 0),
])
def test_bash_has_the_same_compound_and_short_circuit_rules_as_and_lists(runner, commands, stdout, exit_code):
    result = runner.execute_bash(commands)
    assert result["stdout"] == stdout and result["exit_code"] == exit_code


@pytest.mark.parametrize("method,payload,expected_argv", [
    ("python", {"code": 'print("private source café")\n'}, ["python3", "-B", "-u", "-"]),
    ("bash", {"commands": ["printf private-source", "printf second"]}, ["bash", "--noprofile", "--norc", "-s"]),
])
def test_source_is_only_on_stdin_and_preview_never_launches(runner, monkeypatch, method, payload, expected_argv):
    captured = []
    monkeypatch.setattr(shell, "_run_bounded", lambda *args, **kwargs: captured.append((args, kwargs)))
    header, _, source = getattr(runner, f"preview_{method}")(**payload, timeout_seconds=37).partition("\n")
    assert str(runner.root) in header and "37s" in header
    assert captured == []
    getattr(runner, f"execute_{method}")(**payload, timeout_seconds=37)
    (argv, stdin, cwd, env, timeout), kwargs = captured[0]
    assert argv == expected_argv and all("private" not in arg for arg in argv)
    assert stdin == source.encode() and cwd == runner.root and timeout == 37
    if method == "python":
        assert source == payload["code"]
    assert env["HISTFILE"] == "/dev/null" and env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert kwargs == {"stop_on_output_limit": False}
    assert list(runner.root.iterdir()) == []


@pytest.mark.parametrize("method,payload", [
    ("python", {"code": ""}), ("python", {"code": " "}), ("python", {"code": 3}),
    ("python", {"code": "\0"}), ("python", {"code": "é" * 4001}),
    ("bash", {"commands": "pwd"}), ("bash", {"commands": []}), ("bash", {"commands": ("pwd",)}),
    ("bash", {"commands": [""]}), ("bash", {"commands": [" "]}), ("bash", {"commands": ["pwd", 3]}),
    ("bash", {"commands": ["pwd", "\0"]}), ("bash", {"commands": ["x" * 4000, "y" * 4001]}),
])
def test_invalid_sources_fail_during_preview_and_execution(runner, monkeypatch, method, payload):
    monkeypatch.setattr(shell, "_run_bounded", lambda *a, **kw: pytest.fail("must not launch"))
    for operation in ("preview", "execute"):
        with pytest.raises(ValueError):
            getattr(runner, f"{operation}_{method}")(**payload)


@pytest.mark.parametrize("kwargs", [
    {"timeout_seconds": 0}, {"timeout_seconds": 601}, {"timeout_seconds": True}, {"timeout_seconds": 1.5},
    {"cwd": "\0"}, {"cwd": 3}, {"cwd": "missing"},
])
def test_invalid_execution_settings_fail_before_approval(runner, monkeypatch, kwargs):
    monkeypatch.setattr(shell, "_run_bounded", lambda *a, **kw: pytest.fail("must not launch"))
    for method, payload in (("python", {"code": "print(1)"}), ("bash", {"commands": ["pwd"]})):
        for operation in ("preview", "execute"):
            with pytest.raises((ValueError, FileNotFoundError)):
                getattr(runner, f"{operation}_{method}")(**payload, **kwargs)


def test_bash_size_cap_counts_source_not_generated_braces(runner, monkeypatch):
    captured = []
    monkeypatch.setattr(shell, "_run_bounded", lambda *args, **kw: captured.append(args))
    runner.execute_bash(["x" * 4000, "y" * 4000], timeout_seconds=600)
    assert len(captured[0][1]) > shell.MAX_COMMAND_BYTES and captured[0][-1] == 600


def test_local_code_output_is_bounded_and_timeouts_stop_it(runner):
    result = runner.execute_python("import sys; sys.stdout.write('x' * 40000); sys.exit(3)")
    assert result["exit_code"] == 3 and result["truncated"] and not result["timed_out"]
    assert len(result["stdout"].encode()) == shell.MAX_OUTPUT_BYTES
    result = runner.execute_python("import time; time.sleep(30)", timeout_seconds=1)
    assert result["timed_out"]
    result = runner.execute_bash(["printf before", "sleep 30", "printf after"], timeout_seconds=1)
    assert result["timed_out"] and result["stdout"] == "before"


def test_native_code_process_is_reaped_when_interrupted(runner, monkeypatch):
    processes = []
    popen = subprocess.Popen
    selector_type = shell.selectors.DefaultSelector

    def launch(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    def selector():
        instance = selector_type()

        def interrupt(*args):
            raise KeyboardInterrupt

        instance.select = interrupt
        return instance

    monkeypatch.setattr(shell.subprocess, "Popen", launch)
    monkeypatch.setattr(shell.selectors, "DefaultSelector", selector)
    with pytest.raises(KeyboardInterrupt):
        runner.execute_python("import time; time.sleep(30)")
    assert len(processes) == 1 and processes[0].poll() is not None


def test_native_code_tools_have_no_read_only_runner():
    for method in ("run_python", "run_bash", "execute_python", "execute_bash"):
        assert not hasattr(shell.ShellSandbox, method)
