import io
import json

import pytest
from rich.console import Console

from lethe import repl, tools
from lethe.profiles import Profile
from lethe.shell import ShellMode, ShellSandbox


def make_tool(fn=lambda **kw: "result", **kwargs):
    return tools.Tool("run_shell", "test", {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"],
    }, fn, requires_confirmation=True, **kwargs)


def make_config(**kwargs):
    return repl.ReplConfig("http://localhost", None, 4096, "test-model", Profile("test", {}, {}), **kwargs)


def test_streamed_control_sequences_cannot_spoof_the_terminal():
    chunks = ["hello\n\t", "\x1b", "[8mhidden\x1b]52;c;clipboard\x07", "\x9b2J\r\u202ereversed"]
    shown = "".join(repl._terminal_safe(chunk) for chunk in chunks)
    assert shown.startswith("hello\n\t")
    assert "\\x1b[8mhidden" in shown and "\\x1b]52;c;clipboard\\x07" in shown
    assert not any(control in shown for control in ("\x1b", "\x07", "\x9b", "\r", "\u202e"))


def test_registry_opt_in_and_web_composition(tmp_path, monkeypatch):
    probes = []
    monkeypatch.setattr(ShellSandbox, "check_available", lambda self: probes.append(self.root))
    assert tools.build_registry([]) == {}
    assert set(tools.build_registry(["web"])) == {"web_search", "fetch_url"}
    assert probes == []
    registry = tools.build_registry(["shell", "web"], shell_dir=tmp_path)
    assert set(registry) == {"run_shell", "web_search", "fetch_url"}
    assert probes == [tmp_path.resolve()]
    assert registry["run_shell"].requires_confirmation
    assert str(tmp_path.resolve()) in registry["run_shell"].description
    with pytest.raises(KeyError):
        tools.build_registry(["shell", "bogus"], shell_dir=tmp_path)
    assert len(probes) == 1


def test_full_shell_requires_confirmation_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(ShellSandbox, "check_available", lambda self: pytest.fail("sandbox must not be probed"))
    registry = tools.build_registry(["shell", "web"], shell_dir=tmp_path, shell_mode=ShellMode.FULL)
    tool = registry["run_shell"]
    assert set(registry) == {"run_shell", "web_search", "fetch_url"}
    assert tool.shell_mode == ShellMode.FULL and tool.requires_confirmation
    assert "files writable" in tool.confirmation_context and "network allowed" in tool.confirmation_context
    assert tool.parameters["properties"]["timeout_seconds"]["maximum"] == 600
    assert "shell_mode" not in tool.parameters["properties"]
    cfg = make_config(tools=registry)
    output = io.StringIO()
    monkeypatch.setattr(repl, "con", Console(file=output, color_system=None, width=1000))
    args = '{"command":"printf written > result.txt; cat result.txt"}'
    monkeypatch.setattr(repl.Confirm, "ask", lambda *a, **kw: False)
    assert "declined" in repl._invoke_tool(cfg, "run_shell", args)
    assert not (tmp_path / "result.txt").exists()
    monkeypatch.setattr(repl.Confirm, "ask", lambda *a, **kw: True)
    result = json.loads(repl._invoke_tool(cfg, "run_shell", args))
    assert result["exit_code"] == 0 and result["stdout"] == "written"
    assert (tmp_path / "result.txt").read_text() == "written"
    assert "Full access" in output.getvalue()


def test_registry_rejects_unused_or_unknown_shell_mode():
    with pytest.raises(ValueError, match="requires --tools or --shell"):
        tools.build_registry(["web"], shell_mode=ShellMode.FULL)
    with pytest.raises(ValueError):
        tools.build_registry(["shell"], shell_mode="invalid")


@pytest.mark.parametrize("arguments", ['[]', 'null', '"text"', '{', '{}', '{"command":1}', '{"command":"ls","extra":true}'])
def test_invalid_arguments_never_prompt_or_execute(arguments, monkeypatch):
    cfg = make_config(tools={"run_shell": make_tool(lambda **kw: pytest.fail("executed"))})
    monkeypatch.setattr(repl.Confirm, "ask", lambda *a, **kw: pytest.fail("prompted"))
    assert repl._invoke_tool(cfg, "run_shell", arguments).startswith("invalid tool arguments:")


def test_shell_confirmation_is_mandatory_and_preview_is_complete(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(repl, "con", Console(file=output, color_system=None, width=1000))
    executed = []
    cfg = make_config(tools={"run_shell": make_tool(lambda **kw: executed.append(kw) or "ok")})
    command = "printf '" + "x" * 200 + "[conceal] tail \x1b[2J'"
    arguments = json.dumps({"command": command})
    monkeypatch.setattr(repl.Confirm, "ask", lambda *a, **kw: False)
    assert "declined" in repl._invoke_tool(cfg, "run_shell", arguments)
    assert executed == []
    rendered = output.getvalue()
    assert "[conceal] tail" in rendered and "\\u001b[2J" in rendered
    assert "\x1b" not in rendered and "…" not in rendered
    monkeypatch.setattr(repl.Confirm, "ask", lambda *a, **kw: True)
    assert repl._invoke_tool(cfg, "run_shell", arguments) == "ok"
    assert executed == [{"command": command}]


def test_eof_declines_and_disabled_tool_never_runs(monkeypatch):
    def eof(*args, **kwargs):
        raise EOFError
    monkeypatch.setattr(repl.Confirm, "ask", eof)
    cfg = make_config(tools={"run_shell": make_tool(lambda **kw: pytest.fail("executed"))})
    assert "declined" in repl._invoke_tool(cfg, "run_shell", '{"command":"ls"}')
    cfg.tools.clear()
    assert "disabled tool" in repl._invoke_tool(cfg, "run_shell", '{"command":"ls"}')


def test_prompt_tracks_current_tools_and_preserves_custom_prompts():
    base = "This conversation happens in a vacuum: inference is on-device, and nothing is " + tools.VACUUM_CLAIM + "logged." + tools.TOOL_ACCESS_NOTE
    shell_only = tools.sync_system_text(base, {"run_shell": make_tool()})
    assert tools.VACUUM_CLAIM in shell_only and tools.SHELL_NOTE in shell_only
    assert tools.SYSTEM_NOTE not in shell_only
    both = tools.sync_system_text(shell_only, {"run_shell": make_tool(), "web_search": object()})
    assert tools.VACUUM_CLAIM not in both and tools.SYSTEM_NOTE in both
    assert tools.sync_system_text(both, {}) == base  # resumed with tools disabled
    assert tools.sync_system_text(both, {"web_search": object()}).count(tools.SYSTEM_NOTE) == 1
    assert tools.sync_system_text("custom instructions", {"run_shell": make_tool()}) == "custom instructions"


def test_prompt_tracks_full_shell_privacy_and_restores_it_when_disabled():
    base = (
        "This conversation happens in a vacuum: inference is on-device, and nothing is "
        + tools.VACUUM_CLAIM + tools.PRIVACY_NOTE + tools.TOOL_ACCESS_NOTE
    )
    registry = {"run_shell": make_tool(shell_mode=ShellMode.FULL)}
    full = tools.sync_system_text(base, registry)
    assert tools.FULL_SHELL_NOTE in full and tools.FULL_SHELL_PRIVACY_NOTE in full
    assert tools.VACUUM_CLAIM not in full and tools.PRIVACY_NOTE not in full
    assert tools.SHELL_NOTE not in full and tools.SYSTEM_NOTE not in full
    assert tools.sync_system_text(full, registry) == full
    both = tools.sync_system_text(full, {**registry, "web_search": object()})
    assert tools.SYSTEM_NOTE in both and tools.FULL_SHELL_NOTE in both
    assert tools.sync_system_text(both, registry) == full  # /web off keeps shell networking
    readonly = tools.sync_system_text(full, {"run_shell": make_tool()})
    assert tools.SHELL_NOTE in readonly and tools.VACUUM_CLAIM in readonly
    assert tools.FULL_SHELL_NOTE not in readonly and tools.FULL_SHELL_PRIVACY_NOTE not in readonly
    assert tools.sync_system_text(full, {}) == base  # /shell off or resume without tools
    web_only = tools.sync_system_text(full, {"web_search": object()})
    assert tools.FULL_SHELL_NOTE not in web_only and tools.SYSTEM_NOTE in web_only
    assert tools.VACUUM_CLAIM not in web_only
    assert tools.sync_system_text("custom instructions", registry) == "custom instructions"


def test_toolless_builtin_prompt_explains_how_to_request_access():
    legacy = "This conversation happens in a vacuum: inference is on-device, and nothing is " + tools.VACUUM_CLAIM + tools.PRIVACY_NOTE
    current = tools.sync_system_text(legacy, {})
    assert tools.TOOL_ACCESS_NOTE in current and "/tools full" in current
    assert tools.VACUUM_CLAIM in current
    assert tools.sync_system_text(current, {}) == current
    assert tools.sync_system_text("custom instructions", {}) == "custom instructions"


@pytest.mark.parametrize("mode", list(ShellMode))
def test_web_toggle_preserves_shell_and_shell_off_preserves_web(monkeypatch, mode):
    inputs = iter(["/web", "/web off", "/web", "/shell off", "/quit"])
    cfg = make_config(tools={"run_shell": make_tool(shell_mode=mode)})
    observed = []
    def read(prompt):
        observed.append(set(cfg.tools))
        return next(inputs)
    monkeypatch.setattr(repl, "_make_reader", lambda: read)
    repl.run(cfg)
    assert observed == [
        {"run_shell"}, {"run_shell", "web_search", "fetch_url"}, {"run_shell"},
        {"run_shell", "web_search", "fetch_url"}, {"web_search", "fetch_url"},
    ]


def test_full_shell_banner_and_status_describe_current_access(tmp_path, monkeypatch):
    cfg = make_config(tools=tools.build_registry(["shell"], shell_dir=tmp_path, shell_mode=ShellMode.FULL))
    inputs = iter(["/shell", "/quit"])
    monkeypatch.setattr(repl, "_make_reader", lambda: lambda prompt: next(inputs))
    output = io.StringIO()
    monkeypatch.setattr(repl, "con", Console(file=output, color_system=None, width=1000))
    repl.run(cfg)
    shown = output.getvalue()
    assert "full access, files writable, network allowed, inherited environment" in shown
    assert "confirmation required for every command" in shown
    assert "starting directory is not an access boundary" in shown
    assert "read-only" not in shown and "network blocked" not in shown


def test_toolless_chat_can_enable_full_access_then_execute_without_losing_history(tmp_path, monkeypatch):
    root = tmp_path / "project with spaces"
    root.mkdir()
    base = "This conversation happens in a vacuum: inference is on-device, and nothing is " + tools.VACUUM_CLAIM + tools.PRIVACY_NOTE
    cfg = make_config(system=base, shell_dir=tmp_path)
    inputs = iter(["remember the project", f'/tools full "{root}"', "write the file", "/tools off", "status", "/quit"])
    monkeypatch.setattr(repl, "_make_reader", lambda: lambda prompt: next(inputs))
    approvals = []
    def approve(*args, **kwargs):
        approvals.append(args)
        return True
    monkeypatch.setattr(repl.Confirm, "ask", approve)
    calls = []
    def stream(url, messages, **kwargs):
        calls.append(list(messages))
        if len(calls) == 1:
            assert kwargs["tools"] is None
            assert tools.TOOL_ACCESS_NOTE in messages[0]["content"]
            yield repl.Delta(content="I remember the project. You can enable access with /tools full.")
        elif len(calls) == 2:
            assert {t["function"]["name"] for t in kwargs["tools"]} == {"run_shell", "web_search", "fetch_url"}
            assert messages[1]["content"] == "remember the project"
            assert messages[2]["content"].startswith("I remember the project.")
            assert tools.FULL_SHELL_NOTE in messages[0]["content"]
            assert tools.SYSTEM_NOTE in messages[0]["content"]
            assert tools.VACUUM_CLAIM not in messages[0]["content"]
            assert not approvals  # the typed command grants access directly
            yield repl.Delta(tool_calls=[{"id": "write", "type": "function", "function": {
                "name": "run_shell", "arguments": '{"command":"printf enabled > result.txt; cat result.txt"}',
            }}])
        elif len(calls) == 3:
            assert json.loads(messages[-1]["content"])["stdout"] == "enabled"
            yield repl.Delta(content="The file is written.")
        else:
            assert len(calls) == 4 and kwargs["tools"] is None
            assert tools.VACUUM_CLAIM in messages[0]["content"]
            assert tools.FULL_SHELL_NOTE not in messages[0]["content"]
            yield repl.Delta(content="Tools are off.")
    monkeypatch.setattr(repl, "stream_chat", stream)
    repl.run(cfg)
    assert len(calls) == 4 and len(approvals) == 1
    assert (root / "result.txt").read_text() == "enabled"
    assert cfg.shell_dir == root.resolve() and cfg.tools == {}


def test_midchat_shell_mode_switches_keep_directory_web_tools_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr(ShellSandbox, "check_available", lambda self: None)
    cfg = make_config(shell_dir=tmp_path, tools=tools.build_registry(["web"]))
    web = cfg.tools["web_search"]
    messages = [{"role": "user", "content": "earlier message"}, {"role": "assistant", "content": "earlier reply"}]
    history = list(messages)
    for arg, expected_mode in [
        ("read-only", ShellMode.READ_ONLY), ("full", ShellMode.FULL),
        ("off", None), ("full", ShellMode.FULL), ("read-only", ShellMode.READ_ONLY),
    ]:
        repl._tool_command("shell", arg, messages, cfg)
        assert cfg.tools["web_search"] is web
        assert cfg.shell_dir == tmp_path.resolve() and messages == history
        if expected_mode is None:
            assert "run_shell" not in cfg.tools
        else:
            assert cfg.tools["run_shell"].shell_mode == expected_mode
            assert str(tmp_path.resolve()) in cfg.tools["run_shell"].description


@pytest.mark.parametrize("cmd,arg", [
    ("shell", "unknown"), ("shell", "full extra path"), ("shell", "off extra"),
    ("tools", "read-only"), ("tools", "full 'unclosed"), ("tools", "off extra"),
])
def test_invalid_tool_commands_leave_access_unchanged(tmp_path, monkeypatch, cmd, arg):
    cfg = make_config(shell_dir=tmp_path, tools={"run_shell": make_tool()})
    original = dict(cfg.tools)
    messages = [{"role": "user", "content": "keep"}]
    monkeypatch.setattr(tools, "build_registry", lambda *a, **kw: pytest.fail("must not build tools"))
    repl._tool_command(cmd, arg, messages, cfg)
    assert cfg.tools == original and cfg.shell_dir == tmp_path
    assert messages == [{"role": "user", "content": "keep"}]


@pytest.mark.parametrize("cmd,arg", [("shell", "read-only"), ("tools", "full")])
def test_failed_midchat_probe_preserves_active_mode_and_directory(tmp_path, monkeypatch, cmd, arg):
    cfg = make_config(shell_dir=tmp_path, tools=tools.build_registry(["shell", "web"], shell_dir=tmp_path, shell_mode=ShellMode.FULL))
    original = dict(cfg.tools)
    messages = [{"role": "system", "content": "custom instructions"}]
    def unavailable(*a, **kw):
        raise RuntimeError("probe failed")
    monkeypatch.setattr(tools, "build_registry", unavailable)
    output = io.StringIO()
    monkeypatch.setattr(repl, "con", Console(file=output, color_system=None, width=1000))
    repl._tool_command(cmd, f'{arg} "{tmp_path / "different"}"', messages, cfg)
    assert cfg.tools == original and cfg.shell_dir == tmp_path
    assert messages == [{"role": "system", "content": "custom instructions"}]
    assert "tools unchanged: probe failed" in output.getvalue()


@pytest.mark.parametrize("batch_size,expected_executions,expected_streams", [(1, 8, 9), (20, 16, 2)])
def test_tool_loop_has_hard_round_and_total_call_limits(monkeypatch, batch_size, expected_executions, expected_streams):
    inputs = iter(["do work", "/quit"])
    monkeypatch.setattr(repl, "_make_reader", lambda: lambda prompt: next(inputs))
    monkeypatch.setattr(repl.Confirm, "ask", lambda *a, **kw: True)
    executed = []
    cfg = make_config(tools={"run_shell": make_tool(lambda **kw: executed.append(kw) or "ok")})
    advertised = []
    def stream(url, messages, **kwargs):
        advertised.append(kwargs["tools"])
        assert len(advertised) <= expected_streams  # malicious model keeps asking after withdrawal
        yield repl.Delta(tool_calls=[{
            "id": f"call-{len(advertised)}-{i}", "type": "function",
            "function": {"name": "run_shell", "arguments": '{"command":"ls"}'},
        } for i in range(batch_size)])
    monkeypatch.setattr(repl, "stream_chat", stream)
    repl.run(cfg)
    assert len(executed) == expected_executions
    assert len(advertised) == expected_streams and advertised[-1] is None


def test_tool_results_return_to_model_before_final_answer(monkeypatch):
    inputs = iter(["list files", "/quit"])
    monkeypatch.setattr(repl, "_make_reader", lambda: lambda prompt: next(inputs))
    monkeypatch.setattr(repl.Confirm, "ask", lambda *a, **kw: True)
    result = '{"exit_code":0,"stdout":"README.md","stderr":"","timed_out":false,"truncated":false}'
    cfg = make_config(tools={"run_shell": make_tool(lambda **kw: result)})
    rounds = []
    def stream(url, messages, **kwargs):
        rounds.append(list(messages))
        if len(rounds) == 1:
            yield repl.Delta(tool_calls=[{"id": "c1", "type": "function", "function": {
                "name": "run_shell", "arguments": '{"command":"ls"}',
            }}])
        else:
            assert messages[-1] == {"role": "tool", "tool_call_id": "c1", "name": "run_shell", "content": result}
            yield repl.Delta(content="Found README.md.")
    monkeypatch.setattr(repl, "stream_chat", stream)
    repl.run(cfg)
    assert len(rounds) == 2


def test_fetch_pins_curl_to_the_guarded_address(monkeypatch):
    # a rebinding host answers public for the guard; curl must never get to do its own lookup
    monkeypatch.setattr(tools.socket, "getaddrinfo", lambda host, port: [(2, 1, 6, "", ("93.184.216.34", 0))])
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return type("P", (), {"returncode": 0, "stdout": b"HTTP/1.1 404 Not Found\r\n\r\n", "stderr": b""})()

    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(ValueError, match="HTTP 404"):
        tools.fetch_url("https://rebind.test/x")
    argv = seen["argv"]
    assert argv[argv.index("--resolve") + 1] == "rebind.test:443:93.184.216.34"

    monkeypatch.setattr(tools.socket, "getaddrinfo", lambda host, port: [(2, 1, 6, "", ("93.184.216.34", 0)), (2, 1, 6, "", ("10.0.0.5", 0))])
    with pytest.raises(ValueError, match="private/local"):
        tools.fetch_url("http://rebind.test/")
