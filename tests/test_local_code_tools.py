"""Local native-code access, approval, and readable rendering; no commands execute."""

import io
import json

import pytest
from rich.console import Console

from lethe import repl, terminal, tools
from lethe.profiles import Profile
from lethe.shell import FullShell, ShellMode, ShellSandbox


PAYLOADS = [
    ("run_python", {"code": 'import json\nprint(json.dumps({"value": "local-code-marker café"}))\n'}),
    ("run_bash", {"commands": ["value='local-code-marker café'", 'printf "%s\\n" "$value"']}),
]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(FullShell, "check_available", lambda self: None)
    monkeypatch.setattr(ShellSandbox, "check_available", lambda self: None)
    output = io.StringIO()
    monkeypatch.setattr(repl, "con", Console(file=output, color_system=None, width=1000))
    cfg = repl.ReplConfig(
        "http://localhost", None, 4096, "test", Profile("test", {}, {}), shell_dir=tmp_path,
        tools=tools.build_registry(["shell"], shell_dir=tmp_path, shell_mode=ShellMode.FULL),
    )
    return cfg, output


@pytest.mark.parametrize("name,payload", PAYLOADS)
@pytest.mark.parametrize("scope,decision,runs", [
    (None, False, False), (None, "inspect the inputs first", False), (None, True, True),
    ("files", False, False), ("all", None, True),
])
def test_native_code_requires_approval_except_for_all(setup, monkeypatch, name, payload, scope, decision, runs):
    cfg, output = setup
    cfg.auto_approve = scope
    executed = []
    monkeypatch.setattr(FullShell, name.replace("run_", "execute_"),
                        lambda self, **kwargs: executed.append(kwargs) or {"stdout": "ok", "exit_code": 0})
    assert name not in tools.FILE_TOOL_NAMES
    assert cfg.tools[name].requires_confirmation

    def approve():
        assert scope != "all"
        assert not executed
        shown = output.getvalue()
        assert "local-code-marker café" in shown and "Full access" in shown
        if name == "run_python":
            assert payload["code"].rstrip() in shown
            assert "\\nprint" not in shown
        return decision

    monkeypatch.setattr(repl, "ask_approval", approve)
    result = repl._invoke_tool(cfg, name, json.dumps(payload))
    assert bool(executed) is runs
    assert "local-code-marker café" in output.getvalue()
    if runs:
        assert json.loads(result)["stdout"] == "ok"
        assert all(executed[0][key] == value for key, value in payload.items())
    else:
        assert ("not run:" if isinstance(decision, str) else "declined") in result


@pytest.mark.parametrize("name,payload", [
    ("run_python", {}), ("run_python", {"code": ""}), ("run_python", {"code": "\0"}),
    ("run_python", {"code": 1}), ("run_python", {"code": "print(1)", "timeout_seconds": True}),
    ("run_python", {"code": "print(1)", "cwd": "missing-directory"}),
    ("run_python", {"code": "print(1)", "host": "oswald"}),
    ("run_bash", {}), ("run_bash", {"commands": "pwd; uname -s"}),
    ("run_bash", {"commands": []}), ("run_bash", {"commands": ["pwd", 1]}),
    ("run_bash", {"commands": [""]}), ("run_bash", {"commands": ["\0"]}),
    ("run_bash", {"commands": ["pwd"], "timeout_seconds": 601}),
    ("run_bash", {"commands": ["pwd"], "cwd": "missing-directory"}),
])
def test_invalid_code_calls_fail_before_prompt_or_execution(setup, monkeypatch, name, payload):
    cfg, _ = setup
    monkeypatch.setattr(repl, "ask_approval", lambda: pytest.fail("invalid call prompted"))
    monkeypatch.setattr(FullShell, name.replace("run_", "execute_"),
                        lambda *a, **kw: pytest.fail("invalid call executed"))
    result = repl._invoke_tool(cfg, name, json.dumps(payload))
    assert result.startswith(("error:", "invalid tool arguments:"))


def test_native_code_tools_are_full_only_and_removed_on_mode_changes(setup):
    cfg, _ = setup
    names = {"run_python", "run_bash"}
    assert names <= cfg.tools.keys()
    assert all(cfg.tools[name].shell_mode == ShellMode.FULL for name in names)
    assert not names.intersection(tools.FILE_TOOL_NAMES)
    for command in ("read-only", "off"):
        repl._tool_command("shell", command, [], cfg)
        assert not names.intersection(cfg.tools)
        for name, payload in PAYLOADS:
            assert "disabled tool" in repl._invoke_tool(cfg, name, json.dumps(payload))
        repl._tool_command("shell", "full", [], cfg)
        assert names <= cfg.tools.keys()
    repl._tool_command("tools", "off", [], cfg)
    assert cfg.tools == {}


@pytest.mark.parametrize("name,payload", PAYLOADS)
def test_native_code_stream_is_decoded_and_not_repeated_before_approval(setup, monkeypatch, name, payload):
    cfg, output = setup
    arguments = json.dumps(payload)
    call = {"id": "c1", "function": {"name": name, "arguments": arguments}}
    monkeypatch.setattr(FullShell, name.replace("run_", "execute_"),
                        lambda *a, **kw: pytest.fail("declined call executed"))
    with terminal.ToolCallStream(terminal.StatusLine(output)) as stream:
        stream.feed([{"index": 0, "id": "c1", "function": {"name": name}}])
        # Split every escape to ensure the actual body reaches the screen.
        for character in arguments:
            stream.feed([{"index": 0, "function": {"arguments": character}}])
    receipt = stream.shown_call(call)
    assert receipt == 1
    shown_before_approval = output.getvalue()
    assert "local-code-marker café" in shown_before_approval
    if name == "run_python":
        assert payload["code"] in shown_before_approval
    else:
        assert all(command in shown_before_approval for command in payload["commands"])

    def approve():
        shown = output.getvalue()
        assert shown == shown_before_approval + f"tool call 1 · {cfg.tools[name].confirmation_context}\n"
        assert shown.count("local-code-marker café") == 1
        return False

    monkeypatch.setattr(repl, "ask_approval", approve)
    assert "declined" in repl._invoke_tool(cfg, name, arguments, streamed_call=receipt)


@pytest.mark.parametrize("name,payload", [("run_python", {"code": ""}), ("run_bash", {"commands": [""]})])
def test_streamed_native_code_still_requires_successful_preview_validation(setup, monkeypatch, name, payload):
    cfg, output = setup
    arguments = json.dumps(payload)
    call = {"id": "c1", "function": {"name": name, "arguments": arguments}}
    with terminal.ToolCallStream(terminal.StatusLine(output)) as stream:
        stream.feed([{"id": "c1", "function": call["function"]}])
    receipt = stream.shown_call(call)
    assert receipt == 1
    monkeypatch.setattr(repl, "ask_approval", lambda: pytest.fail("must validate before approval"))
    monkeypatch.setattr(FullShell, name.replace("run_", "execute_"),
                        lambda *a, **kw: pytest.fail("invalid call executed"))
    assert repl._invoke_tool(cfg, name, arguments, streamed_call=receipt).startswith("error:")
