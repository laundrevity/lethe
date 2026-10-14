"""Opt-in smoke test with the default model, loaded for the test and unloaded after it.

Only a fixed, harmless printf command can execute. Prompts/output stay in memory,
and the real read-only sandbox is used. Run from outside any outer sandbox:
    uv run pytest -q -s tests/test_live_tools.py --run-live-tools
"""

import json

import pytest

from lethe import cli, config, repl, spec, tools
from lethe.chat import Delta, TurnStats, stream_chat
from lethe.profiles import profile_for


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
