import io
import json

import pytest
from rich.console import Console

from lethe import repl, tools
from lethe.profiles import Profile
from lethe.shell import ShellMode, ShellSandbox
from lethe.ssh import SSHRunner


def config(tmp_path, **kwargs):
    return repl.ReplConfig(
        "http://localhost", None, 4096, "test", Profile("test", {}, {}), shell_dir=tmp_path,
        tools=tools.build_registry(["shell"], shell_dir=tmp_path, shell_mode=ShellMode.FULL), **kwargs,
    )


@pytest.mark.parametrize("scope,decision,runs", [
    (None, False, False), (None, "use a different host", False), (None, True, True),
    ("files", False, False), ("all", None, True),
])
def test_ssh_approval_covers_host_and_source(tmp_path, monkeypatch, scope, decision, runs):
    cfg = config(tmp_path, auto_approve=scope)
    output = io.StringIO()
    monkeypatch.setattr(repl, "con", Console(file=output, color_system=None, width=1000))
    executed = []
    monkeypatch.setattr(SSHRunner, "execute", lambda self, **kwargs: executed.append(kwargs) or {"stdout": "ok"})
    source = 'import json\nprint(json.dumps({"message": "hello"}))\n'

    def approve():
        assert scope != "all"
        shown = output.getvalue()
        assert "oswald" in shown and "remote project" in shown
        assert source.rstrip() in shown and "\\nprint" not in shown
        assert "33" in shown
        return decision

    monkeypatch.setattr(repl, "ask_approval", approve)
    arguments = {"host": "oswald", "python": source, "cwd": "~/remote project", "timeout_seconds": 33}
    result = repl._invoke_tool(cfg, "run_ssh", json.dumps(arguments))
    assert bool(executed) is runs
    if runs:
        assert json.loads(result)["stdout"] == "ok"
        assert executed[0]["host"] == "oswald" and executed[0]["python"] == source
        assert "oswald" in output.getvalue() and source.rstrip() in output.getvalue()
    else:
        assert "not run:" in result if isinstance(decision, str) else "declined" in result


@pytest.mark.parametrize("payload", [
    {}, {"bash": []}, {"bash": ["pwd", 3]}, {"bash": {"command": "pwd"}},
    {"bash": [""]}, {"bash": "pwd"}, {"bash": ["pwd"], "python": "print(1)"}, {"python": None},
    {"python": ""}, {"python": "print(1)", "timeout_seconds": True},
    {"python": "print(1)", "timeout_seconds": 601},
    {"python": "print(1)", "host": "-oProxyCommand=bad"},
    {"python": "print(1)", "extra": "ignored?"},
])
def test_invalid_ssh_never_prompts_or_executes(tmp_path, monkeypatch, payload):
    cfg = config(tmp_path)
    monkeypatch.setattr(repl, "ask_approval", lambda: pytest.fail("must not prompt"))
    monkeypatch.setattr(SSHRunner, "execute", lambda *a, **kw: pytest.fail("must not run"))
    result = repl._invoke_tool(cfg, "run_ssh", json.dumps({"host": "oswald", **payload}))
    assert result.startswith(("error:", "invalid tool arguments:"))


def test_ssh_is_removed_with_shell_and_never_part_of_file_autoapproval(tmp_path, monkeypatch):
    monkeypatch.setattr(ShellSandbox, "check_available", lambda self: None)
    cfg = config(tmp_path)
    assert "run_ssh" not in tools.FILE_TOOL_NAMES
    assert cfg.tools["run_ssh"].requires_confirmation
    for command in ("read-only", "off"):
        repl._tool_command("shell", command, [], cfg)
        assert "run_ssh" not in cfg.tools
        assert "disabled tool" in repl._invoke_tool(cfg, "run_ssh", '{"host":"oswald","bash":["pwd"]}')
        repl._tool_command("shell", "full", [], cfg)
        assert "run_ssh" in cfg.tools
    repl._tool_command("tools", "off", [], cfg)
    assert not cfg.tools


def test_ssh_prompt_tracks_access_without_touching_user_instructions(tmp_path, monkeypatch):
    from lethe.instructions import MARKER

    monkeypatch.setattr(ShellSandbox, "check_available", lambda self: None)
    base = "This conversation happens in a vacuum: inference is on-device, and nothing is " + tools.VACUUM_CLAIM + tools.PRIVACY_NOTE
    instructions = MARKER + "Keep this project instruction."
    registry = config(tmp_path).tools
    full = tools.sync_system_text(base + instructions, registry)
    assert full.count(tools.SSH_NOTE) == 1 and full.endswith(instructions)
    assert tools.sync_system_text(full, registry) == full
    legacy = base + tools._LEGACY_SSH_NOTE + instructions
    updated = tools.sync_system_text(legacy, registry)
    assert tools._LEGACY_SSH_NOTE not in updated and updated.count(tools.SSH_NOTE) == 1
    assert updated.endswith(instructions)
    readonly = tools.build_registry(["shell"], shell_dir=tmp_path)
    assert tools.SSH_NOTE not in tools.sync_system_text(full, readonly)
    assert tools.SSH_NOTE not in tools.sync_system_text(full, {})
