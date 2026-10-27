"""REPL batching/approval/history integration, with no worker processes or network."""

import copy
import io
import json
import threading
from types import SimpleNamespace

import pytest
from rich.console import Console

from lethe import repl, tools, webbatch
from lethe.profiles import Profile


def call(number, name="web_search", **arguments):
    return {"id": f"c{number}", "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}


def request_value(request):
    arguments = request["arguments"]
    return arguments.get("query", arguments.get("url", arguments.get("command")))


def registry(events):
    result = {}
    for name, parameter in (("web_search", "query"), ("fetch_url", "url"), ("run_shell", "command")):
        def execute(_name=name, **arguments):
            value = next(iter(arguments.values()))
            events.append(("serial", _name, value))
            return f"result:{value}"

        result[name] = tools.Tool(
            name, "test", {"type": "object", "properties": {parameter: {"type": "string"}},
                           "required": [parameter]}, execute,
            requires_confirmation=name == "run_shell", shell_mode=tools.ShellMode.FULL if name == "run_shell" else None,
        )
    return result


def config(events, **kwargs):
    return repl.ReplConfig("http://localhost", None, 4096, "test", Profile("test", {}, {}),
                           tools=registry(events), **kwargs)


@pytest.fixture
def output(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(repl.sys, "stdout", stream)
    monkeypatch.setattr(repl, "con", Console(file=stream, color_system=None, width=1000))
    return stream


def install_batch(monkeypatch, events, *, before_start=None):
    batches = []

    class FakeBatch:
        def __init__(self, requests, searx=None):
            if before_start:
                before_start(requests, searx)
            self.requests, self.results, self.started = requests, {}, set()
            batches.append(self)
            events.append(("batch", [request_value(request) for request in requests]))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            events.append(("closed",))

        def __iter__(self):
            self.started.update(range(len(self.requests)))
            for index in reversed(range(len(self.requests))):
                value = request_value(self.requests[index])
                self.results[index] = f"result:{value}"
                events.append(("completed", value))
                yield index, self.results[index]

    monkeypatch.setattr(webbatch, "WebBatch", FakeBatch)
    return batches


def drive(monkeypatch, cfg, calls, *, after_interrupt=False):
    inputs = iter(["look things up", *(["continue"] if after_interrupt else []), "/quit"])
    monkeypatch.setattr(repl, "_make_reader", lambda: lambda prompt: next(inputs))
    seen = []

    def stream(url, messages, **kwargs):
        seen.append((copy.deepcopy(messages), kwargs))
        if len(seen) == 1:
            yield repl.Delta(content="Looking it up.", reasoning="Need independent sources.")
            yield repl.Delta(tool_calls=calls)
        else:
            yield repl.Delta(content="Done.")
        yield repl.TurnStats()

    monkeypatch.setattr(repl, "stream_chat", stream)
    repl.run(cfg)
    return seen


def test_contiguous_web_calls_batch_around_serial_barriers_and_history_stays_ordered(monkeypatch, output):
    events = []
    batches = install_batch(monkeypatch, events)
    calls = [
        call(1, query="first"), call(2, "fetch_url", url="https://two.invalid"),
        call(3, "run_shell", command="local step"),
        call(4, query="fourth"), call(5, "fetch_url", url="https://five.invalid"),
        call(6, "run_shell", command="another local step"), call(7, query="last singleton"),
    ]
    seen = drive(monkeypatch, config(events, auto_approve="all"), calls)
    assert len(batches) == 2 and len(seen) == 2
    assert events == [
        ("batch", ["first", "https://two.invalid"]),
        ("completed", "https://two.invalid"), ("completed", "first"), ("closed",),
        ("serial", "run_shell", "local step"),
        ("batch", ["fourth", "https://five.invalid"]),
        ("completed", "https://five.invalid"), ("completed", "fourth"), ("closed",),
        ("serial", "run_shell", "another local step"), ("serial", "web_search", "last singleton"),
    ]
    results = [message for message in seen[1][0] if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in results] == [f"c{i}" for i in range(1, 8)]
    assert [message["content"] for message in results] == [
        "result:first", "result:https://two.invalid", "result:local step", "result:fourth",
        "result:https://five.invalid", "result:another local step", "result:last singleton",
    ]


@pytest.mark.parametrize("auto_approve", [None, "files"])
def test_every_preview_and_confirmation_precedes_launch_and_excludes_declines_notes_errors(monkeypatch, output, auto_approve):
    events, recorded, approvals = [], [], []
    cfg = config(events, tools_confirm=True, auto_approve=auto_approve, searx="https://search.invalid")
    calls = [call(1, query="decline-me"), call(2, query="ask-again"), call(3),
             call(4, query="approved-query"), call(5, "fetch_url", url="https://approved.invalid")]
    decisions = iter([False, "make the query more specific", True, True])
    main_thread = threading.get_ident()

    def approve():
        assert threading.get_ident() == main_thread
        assert not events  # no workers or direct web calls started while asking
        approvals.append(True)
        return next(decisions)

    def before_start(requests, searx):
        assert threading.get_ident() == main_thread and len(approvals) == 4
        assert searx == cfg.searx
        assert [request_value(request) for request in requests] == ["approved-query", "https://approved.invalid"]
        assert all(value in output.getvalue() for value in ("decline-me", "ask-again", "approved-query", "https://approved.invalid"))

    monkeypatch.setattr(repl, "ask_approval", approve)
    batches = install_batch(monkeypatch, events, before_start=before_start)
    repl._invoke_web_group(cfg, calls, SimpleNamespace(shown_call=lambda call: None), allowance=5,
                           record_result=lambda call, result: recorded.append((call["id"], result)),
                           show_status=lambda phase: None)
    assert len(batches) == 1 and [item[0] for item in recorded] == [f"c{i}" for i in range(1, 6)]
    assert "declined" in recorded[0][1]
    assert "not run:" in recorded[1][1] and "make the query more specific" in recorded[1][1]
    assert recorded[2][1].startswith("invalid tool arguments:")
    assert [item[1] for item in recorded[3:]] == ["result:approved-query", "result:https://approved.invalid"]


def test_web_batch_respects_total_call_allowance_across_later_tools(monkeypatch, output):
    events = []
    batches = install_batch(monkeypatch, events)
    monkeypatch.setattr(repl, "MAX_TOOL_CALLS", 2)
    monkeypatch.setattr(repl, "ask_approval", lambda: pytest.fail("auto-approve all should not prompt"))
    calls = [call(1, query="first"), call(2, query="second"), call(3, query="over limit"),
             call(4, "run_shell", command="not run"), call(5, query="also over"), call(6, query="still over")]
    seen = drive(monkeypatch, config(events, tools_confirm=True, auto_approve="all"), calls)
    assert len(batches) == 1
    assert [request_value(request) for request in batches[0].requests] == ["first", "second"]
    assert not any(event[0] == "serial" for event in events)
    results = [message for message in seen[1][0] if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in results] == [f"c{i}" for i in range(1, 7)]
    assert [message["content"] for message in results[:2]] == ["result:first", "result:second"]
    assert all("tool-call limit reached" in message["content"] for message in results[2:])
    assert seen[1][1]["tools"] is None


def test_interrupt_during_confirmation_launches_nothing_and_records_every_call(monkeypatch, output):
    events, recorded = [], []
    cfg = config(events, tools_confirm=True)
    attempts = 0

    def approve():
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise KeyboardInterrupt
        return True

    monkeypatch.setattr(repl, "ask_approval", approve)
    monkeypatch.setattr(webbatch, "WebBatch", lambda *a, **kw: pytest.fail("must not launch before approvals finish"))
    calls = [call(1, query="approved but not sent"), call(2, query="interrupt here"), call(3, query="never asked")]
    with pytest.raises(KeyboardInterrupt):
        repl._invoke_web_group(cfg, calls, SimpleNamespace(shown_call=lambda call: None), allowance=3,
                               record_result=lambda call, result: recorded.append((call["id"], result)),
                               show_status=lambda phase: None)
    assert attempts == 2 and events == []
    assert recorded == [(f"c{i}", "not run: interrupted by the user") for i in range(1, 4)]


def test_interrupt_keeps_close_time_results_and_distinguishes_sent_from_queued(monkeypatch, output):
    events = []

    class InterruptedBatch:
        def __init__(self, requests, searx=None):
            self.requests, self.results, self.started = requests, {}, set()
            assert [request_value(request) for request in requests] == ["close-result", "yielded-result", "in-flight", "queued"]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.results[0] = "completed while closing"

        def __iter__(self):
            self.started.update({0, 1, 2})
            self.results[1] = "completed before interrupt"
            yield 1, self.results[1]
            raise KeyboardInterrupt

    monkeypatch.setattr(webbatch, "WebBatch", InterruptedBatch)
    decisions = iter([False, True, True, True, True])
    monkeypatch.setattr(repl, "ask_approval", lambda: next(decisions))
    calls = [call(i, query=value) for i, value in enumerate(
        ["declined", "close-result", "yielded-result", "in-flight", "queued"], 1)]
    seen = drive(monkeypatch, config(events, tools_confirm=True), calls, after_interrupt=True)
    assert len(seen) == 2 and events == []
    kept = seen[1][0]
    results = [message for message in kept if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in results] == [f"c{i}" for i in range(1, 6)]
    assert "declined" in results[0]["content"]
    assert results[1]["content"] == "completed while closing"
    assert results[2]["content"] == "completed before interrupt"
    assert "may have been sent" in results[3]["content"]
    assert results[4]["content"] == "not run: interrupted by the user"
    assert kept[-1]["role"] == "user" and kept[-1]["content"].startswith(repl.INTERRUPT_NOTE)
    # The group recorded every tool reply, leaving awaiting empty. The original
    # assistant content/reasoning must not be appended a second time on Ctrl-C.
    assistants = [message for message in kept if message["role"] == "assistant"]
    assert len(assistants) == 2
    assert assistants[0]["content"] == "Looking it up." and assistants[0]["reasoning_content"] == "Need independent sources."
    assert assistants[0]["tool_calls"] == calls
    assert assistants[1] == {"role": "assistant", "content": repl.NO_OUTPUT}
