"""The streamed call is the approval body; completion adds only missing details."""

import io
import json

import pytest
from rich.console import Console

from lethe import repl, terminal, tools
from lethe.profiles import Profile
from lethe.shell import FullShell, ShellMode
from lethe.ssh import SSHRunner


@pytest.fixture
def screen(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(repl, "con", Console(file=output, color_system=None, width=1000))
    return output


def config(tmp_path, **kwargs):
    return repl.ReplConfig(
        "http://localhost", None, 4096, "test", Profile("test", {}, {}),
        tools=tools.build_registry(["shell"], shell_dir=tmp_path, shell_mode=ShellMode.FULL), **kwargs,
    )


def display(output, name, payload):
    call = {"id": "c1", "function": {"name": name, "arguments": json.dumps(payload)}}
    with terminal.ToolCallStream(terminal.StatusLine(output)) as stream:
        arguments = call["function"]["arguments"]
        stream.feed([{"index": 0, "id": "c1", "function": {"name": name, "arguments": arguments[:20]}}])
        stream.feed([{"index": 0, "function": {"arguments": arguments[20:]}}])
    return call, stream.shown_call(call)


@pytest.mark.parametrize("auto,decision,runs", [(None, False, False), (None, "use another command", False),
                                               (None, True, True), ("all", None, True)])
def test_streamed_shell_shown_once_with_approval_context(tmp_path, monkeypatch, screen, auto, decision, runs):
    cfg = config(tmp_path, auto_approve=auto)
    executed = []
    monkeypatch.setattr(FullShell, "execute", lambda self, *a, **kw: executed.append((a, kw)) or {"stdout": "done"})
    payload = {"command": "printf unique-command"}
    call, receipt = display(screen, "run_shell", payload)
    assert receipt == 1

    def approve():
        assert auto != "all"
        shown = screen.getvalue()
        assert shown.count("printf unique-command") == 1
        assert "tool call 1 · Full access" in shown
        assert "network allowed" in shown and str(tmp_path) in shown
        return decision

    monkeypatch.setattr(repl, "ask_approval", approve)
    result = repl._invoke_tool(cfg, "run_shell", call["function"]["arguments"], streamed_call=receipt)
    assert bool(executed) is runs
    assert screen.getvalue().count("printf unique-command") == 1
    assert "⚙ 'run_shell'" not in screen.getvalue()
    if not runs:
        assert ("not run:" if isinstance(decision, str) else "declined") in result


@pytest.mark.parametrize("payload", [
    {"python": 'import json\nprint(json.dumps({"value": "unique-source"}))\n'},
    {"bash": ["cd ~/project", "printf unique-source"]},
    {"bash": ["for value in unique-source; do\n  printf '%s' \"$value\"\ndone"]},
])
def test_streamed_ssh_does_not_repeat_the_preview_header(tmp_path, monkeypatch, screen, payload):
    cfg = config(tmp_path)
    monkeypatch.setattr(SSHRunner, "execute", lambda *a, **kw: pytest.fail("declined SSH must not connect"))
    arguments = {"host": "oswald", **payload}
    call, receipt = display(screen, "run_ssh", arguments)
    assert receipt == 1
    streamed = screen.getvalue()

    def approve():
        shown = screen.getvalue()
        assert shown.count("unique-source") == 1
        assert shown == streamed + "tool call 1 · SSH · full access on the remote host · source sent over the network\n"
        return False

    monkeypatch.setattr(repl, "ask_approval", approve)
    result = repl._invoke_tool(cfg, "run_ssh", call["function"]["arguments"], streamed_call=receipt)
    assert "declined" in result
    assert "⚙ 'run_ssh'" not in screen.getvalue()


def test_streamed_ssh_still_validates_before_approval(tmp_path, monkeypatch, screen):
    cfg = config(tmp_path)
    call, receipt = display(screen, "run_ssh", {"host": "oswald", "bash": ["pwd"], "python": "print(1)"})
    monkeypatch.setattr(repl, "ask_approval", lambda: pytest.fail("must validate first"))
    monkeypatch.setattr(SSHRunner, "execute", lambda *a, **kw: pytest.fail("must not run"))
    assert "exactly one" in repl._invoke_tool(cfg, "run_ssh", call["function"]["arguments"], streamed_call=receipt)


def test_streamed_file_edit_still_shows_the_actual_diff(tmp_path, monkeypatch, screen):
    cfg = config(tmp_path)
    target = tmp_path / "example.py"
    target.write_text("value = 1\n")
    call, receipt = display(screen, "edit_file", {"path": "example.py", "old_string": "value = 1", "new_string": "value = 2"})

    def approve():
        shown = screen.getvalue()
        assert "-value = 1\n+value = 2" in shown
        assert str(target) in shown and "tool call 1 · Full access" in shown
        return False

    monkeypatch.setattr(repl, "ask_approval", approve)
    assert "declined" in repl._invoke_tool(cfg, "edit_file", call["function"]["arguments"], streamed_call=receipt)
    assert target.read_text() == "value = 1\n"
    assert "⚙ 'edit_file'" not in screen.getvalue()


@pytest.mark.parametrize("reason", ["missing", "changed", "interleaved"])
def test_incomplete_or_fragmented_display_gets_full_approval_body(tmp_path, monkeypatch, screen, reason):
    cfg = config(tmp_path)
    arguments = json.dumps({"command": "printf final-command"})
    call = {"id": "c1", "function": {"name": "run_shell", "arguments": arguments}}
    with terminal.ToolCallStream(terminal.StatusLine(screen)) as stream:
        if reason == "changed":
            stream.feed([{"id": "c1", "function": {"name": "run_shell", "arguments": '{"command":"different"}'}}])
        elif reason == "interleaved":
            stream.feed([{"id": "c1", "function": {"name": "run_shell", "arguments": arguments[:20]}}])
            stream.feed([{"index": 1, "id": "c2", "function": {"name": "read_file", "arguments": '{"path":"x"}'}}])
            stream.feed([{"index": 0, "function": {"arguments": arguments[20:]}}])
    assert stream.shown_call(call) is None

    def approve():
        assert "⚙ 'run_shell'" in screen.getvalue()
        assert arguments in screen.getvalue()
        return False

    monkeypatch.setattr(repl, "ask_approval", approve)
    assert "declined" in repl._invoke_tool(cfg, "run_shell", arguments, streamed_call=stream.shown_call(call))
