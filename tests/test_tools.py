import io
import json

import pytest
from rich.console import Console

from lethe import repl, tools
from lethe.profiles import Profile
from lethe.shell import ShellSandbox


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
    base = "This conversation happens in a vacuum: inference is on-device, and nothing is " + tools.VACUUM_CLAIM + "logged."
    shell_only = tools.sync_system_text(base, {"run_shell": make_tool()})
    assert tools.VACUUM_CLAIM in shell_only and tools.SHELL_NOTE in shell_only
    assert tools.SYSTEM_NOTE not in shell_only
    both = tools.sync_system_text(shell_only, {"run_shell": make_tool(), "web_search": object()})
    assert tools.VACUUM_CLAIM not in both and tools.SYSTEM_NOTE in both
    assert tools.sync_system_text(both, {}) == base  # resumed with tools disabled
    assert tools.sync_system_text(both, {"web_search": object()}).count(tools.SYSTEM_NOTE) == 1
    assert tools.sync_system_text("custom instructions", {"run_shell": make_tool()}) == "custom instructions"


def test_web_toggle_preserves_shell_and_shell_off_preserves_web(monkeypatch):
    inputs = iter(["/web", "/web off", "/web", "/shell off", "/quit"])
    cfg = make_config(tools={"run_shell": make_tool()})
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
