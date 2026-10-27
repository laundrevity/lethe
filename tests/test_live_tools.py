"""Opt-in smoke test with the default model, loaded for the test and unloaded after it.

Only a fixed, harmless printf command can execute. Prompts/output stay in memory,
and the real read-only sandbox is used. The SSH smoke test checks generated calls
without ever executing them or connecting to a remote host. Run outside any outer sandbox:
    uv run pytest -q -s tests/test_live_tools.py --run-live-tools
"""

import io
import json

import pytest
from rich.console import Console

from lethe import cli, config, repl, spec, terminal, tools
from lethe.chat import Delta, TurnStats, stream_chat
from lethe.profiles import profile_for
from lethe.ssh import SSHRunner


def test_live_model_shell_roundtrip(request, tmp_path, monkeypatch):
    if not request.config.getoption("--run-live-tools"):
        pytest.skip("live model test: pass --run-live-tools (loads the default model)")
    default = config.default_model()
    assert default, "set a default model first: lethe default owner/repo:QUANT"
    model = spec.resolve(default)
    shape = cli._shape(model.path)
    profile = profile_for(shape)
    registry = tools.build_registry(["shell"], shell_dir=tmp_path)
    with cli._start(model, ctx=16384, kv="f16", vision=False, thinking=True, effort=None) as srv:
        _roundtrip(srv, model, shape, profile, registry, repl.ReplConfig(srv.url, srv.api_key, 0, model.spec, profile, tools=registry), monkeypatch)


def test_live_model_ssh_arguments(request, tmp_path, monkeypatch):
    """Exercise the real template/grammar/model boundary missed by mocked SSE tests."""
    if not request.config.getoption("--run-live-tools"):
        pytest.skip("live model test: pass --run-live-tools (loads the default model)")
    default = config.default_model()
    assert default, "set a default model first: lethe default owner/repo:QUANT"
    model = spec.resolve(default)
    shape = cli._shape(model.path)
    profile = profile_for(shape)
    registry = tools.build_registry(["shell"], shell_dir=tmp_path, shell_mode=tools.ShellMode.FULL)
    monkeypatch.setattr(repl, "ask_approval", lambda: False)
    monkeypatch.setattr(SSHRunner, "execute", lambda *a, **kw: pytest.fail("live SSH smoke must never connect"))
    prompts = [
        ("give me list of all dirs in home dir on oswald", None),
        ("On oswald, show the current directory with pwd, then show the kernel name with uname -s. "
         "Request one SSH call.", "array"),
        ("Use run_ssh on oswald with python source consisting of these two lines:\n"
         "import json\nprint(json.dumps({'status': 'lethe-ssh-smoke'}))\nRequest one call.", "python"),
    ]
    with cli._start(model, ctx=16384, kv="f16", vision=False, thinking=True, effort=None) as srv:
        for prompt, payload_type in prompts:
            messages = [
                {"role": "system", "content": tools.sync_system_text(cli._default_system(model, shape), registry)},
                {"role": "user", "content": prompt},
            ]
            calls = []
            display = io.StringIO()
            with terminal.ToolCallStream(terminal.StatusLine(display)) as rendered:
                for event in stream_chat(
                    srv.url, messages, n_ctx=16384, sampling={**profile.sampling(True), "seed": 0},
                    max_tokens=1024, api_key=srv.api_key, tools=[tool.spec() for tool in registry.values()],
                ):
                    if isinstance(event, Delta):
                        rendered.feed(event.tool_call_deltas or [])
                        calls.extend(event.tool_calls or [])
            assert len(calls) == 1, f"Expected one populated SSH call, got {len(calls)}"
            call = calls[0]["function"]
            assert call["name"] == "run_ssh"
            args = registry["run_ssh"].parse_arguments(call["arguments"])
            assert args["host"] == "oswald"
            if "bash" in args:
                assert isinstance(args["bash"], list) and args["bash"]
            # Exercise the display-to-approval handoff with actual server chunks.
            # Decline every call; never execute source or open an SSH connection.
            receipt = rendered.shown_call(calls[0])
            assert receipt == 1
            monkeypatch.setattr(repl, "con", Console(file=display, color_system=None, width=1000))
            cfg = repl.ReplConfig(srv.url, srv.api_key, 16384, model.spec, profile, tools=registry)
            assert "declined" in repl._invoke_tool(cfg, "run_ssh", call["arguments"], streamed_call=receipt)
            assert display.getvalue().count("⚙") == 1
            assert "tool call 1 · SSH" in display.getvalue()
            assert "host='oswald'" not in display.getvalue()
            assert "timeout=120s" not in display.getvalue()
            if "python" in args:
                assert display.getvalue().count(args["python"]) == 1
            if payload_type == "array":
                assert args["bash"] == ["pwd", "uname -s"]
            elif payload_type == "python":
                assert "import json\n" in args["python"]
                assert "lethe-ssh-smoke" in args["python"]
            print(f"Live SSH arguments passed: {payload_type or 'directory listing'}; no SSH connection made.")


def test_live_model_local_code_and_web_batch(request, tmp_path):
    """Check native source and multiple web calls, without executing any tools."""
    if not request.config.getoption("--run-live-tools"):
        pytest.skip("live model test: pass --run-live-tools (loads the default model)")
    model = spec.resolve(config.default_model())
    shape = cli._shape(model.path)
    profile = profile_for(shape)
    registry = tools.build_registry(["shell", "web"], shell_dir=tmp_path, shell_mode=tools.ShellMode.FULL)
    prompts = [
        ("Use Python locally to execute exactly these two lines:\n"
         "import json\nprint(json.dumps({'status': 'lethe-local-smoke'}))\nRequest one call.", "run_python", 1),
        ("Using local Bash, show the current directory with pwd, then the kernel name with uname -s. "
         "Request one call.", "run_bash", 1),
        ("Perform two independent web searches: one for official Python documentation and one for the official "
         "Bash manual. Request both searches together.", "web_search", 2),
    ]
    with cli._start(model, ctx=16384, kv="f16", vision=False, thinking=True, effort=None) as srv:
        for prompt, expected, count in prompts:
            messages = [
                {"role": "system", "content": tools.sync_system_text(cli._default_system(model, shape), registry)},
                {"role": "user", "content": prompt},
            ]
            calls = []
            for event in stream_chat(
                srv.url, messages, n_ctx=16384, sampling={**profile.sampling(True), "seed": 0},
                max_tokens=1536, api_key=srv.api_key, tools=[tool.spec() for tool in registry.values()],
            ):
                if isinstance(event, Delta):
                    calls.extend(event.tool_calls or [])
            assert len(calls) == count, f"Expected {count} {expected} call(s), got {len(calls)}"
            for call in calls:
                function = call["function"]
                assert function["name"] == expected
                args = registry[expected].parse_arguments(function["arguments"])
                if expected == "run_python":
                    assert "import json\n" in args["code"] and "lethe-local-smoke" in args["code"]
                    registry[expected].preview(**args)
                elif expected == "run_bash":
                    assert args["commands"] == ["pwd", "uname -s"]
                    registry[expected].preview(**args)
                else:
                    assert args["query"].strip()
            print(f"Live tool selection passed: {expected}; no tools executed.")


def _roundtrip(srv, model, shape, profile, registry, cfg, monkeypatch):
    command = "printf lethe-shell-smoke-ok"
    messages = [
        {"role": "system", "content": tools.sync_system_text(cli._default_system(model, shape), registry)},
        {"role": "user", "content": f"Use run_shell to execute exactly `{command}`, then report the actual stdout. Do not simulate the result."},
    ]
    wire_tools = [tool.spec() for tool in registry.values()]

    def turn():
        calls, text, finish = [], [], None
        for event in stream_chat(
            cfg.url, messages, n_ctx=0, sampling=profile.sampling(True),
            max_tokens=1024, api_key=cfg.api_key, tools=wire_tools,
        ):
            if isinstance(event, Delta):
                text.append(event.content)
                calls.extend(event.tool_calls or [])
            elif isinstance(event, TurnStats):
                finish = event.finish_reason
        return calls, "".join(text), finish

    calls, text, finish = turn()
    assert len(calls) == 1, f"Expected one actual tool call, got {len(calls)} (finish={finish}, text={text!r})"
    call = calls[0]
    assert call["function"]["name"] == "run_shell"
    args = registry["run_shell"].parse_arguments(call["function"]["arguments"])
    assert args["command"] == command, "The smoke test will only approve the fixed printf command"
    assert args.get("cwd", ".") == ".", "The smoke test will only approve its isolated workspace"
    # Exercise the real dispatch/approval path after checking the exact command.
    monkeypatch.setattr(repl, "ask_approval", lambda *a, **kw: True)
    result = repl._invoke_tool(cfg, "run_shell", call["function"]["arguments"])
    payload = json.loads(result)
    assert payload["exit_code"] == 0 and payload["stdout"] == "lethe-shell-smoke-ok"
    messages.extend([
        {"role": "assistant", "content": text or None, "tool_calls": calls},
        {"role": "tool", "tool_call_id": call["id"], "name": "run_shell", "content": result},
    ])
    followup_calls, answer, finish = turn()
    assert not followup_calls, "Expected the final answer after the tool result"
    assert "lethe-shell-smoke-ok" in answer, f"Model did not report the tool output (finish={finish})"
    print("Live roundtrip passed: model requested run_shell, sandbox executed printf, model reported the actual output.")
