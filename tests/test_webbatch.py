import io
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from lethe import tools, web_worker, webbatch


FAKE_WORKER = r'''
import json, os, sys, time
request = json.load(sys.stdin)
args = request["arguments"]
started = time.monotonic()
time.sleep(args.get("delay", 0))
mode = args.get("mode", "normal")
if mode == "bad":
    print("not json")
elif mode == "large":
    print("x" * 200000)
elif mode == "exit":
    sys.exit(7)
else:
    result = {
        "value": args.get("value"), "started": started, "finished": time.monotonic(),
        "searx": request.get("searx"), "pid": os.getpid(),
    }
    print(json.dumps(json.dumps(result)), flush=True)
    if mode == "linger":
        time.sleep(30)
'''


@pytest.fixture
def fake_workers(monkeypatch):
    monkeypatch.setattr(webbatch, "_worker_argv", lambda: [sys.executable, "-B", "-c", FAKE_WORKER])
    processes = []
    real_popen = subprocess.Popen

    def launch(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append((process, args, kwargs))
        return process

    monkeypatch.setattr(webbatch.subprocess, "Popen", launch)
    yield processes
    assert all(process.poll() is not None for process, _, _ in processes)


def request(value=0, **arguments):
    return {"name": "web_search", "arguments": {"value": value, **arguments}}


def test_web_jobs_overlap_without_exceeding_four_workers(fake_workers):
    with webbatch.WebBatch([request(i, delay=0.2) for i in range(7)]) as batch:
        results = dict(batch)
    assert set(results) == set(range(7)) and batch.started == set(range(7))
    records = [json.loads(result) for result in results.values()]
    events = sorted([(record["started"], 1) for record in records] + [(record["finished"], -1) for record in records])
    active = peak = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    assert 2 <= peak <= 4
    assert len(fake_workers) == 7


def test_results_arrive_in_completion_order_with_original_indices(fake_workers):
    with webbatch.WebBatch([request("slow", delay=0.3), request("fast")], searx="https://search.example") as batch:
        completed = list(batch)
    assert [index for index, _ in completed] == [1, 0]
    assert json.loads(batch.results[0])["value"] == "slow"
    assert json.loads(batch.results[1])["value"] == "fast"
    assert all(json.loads(result)["searx"] == "https://search.example" for _, result in completed)


def test_requests_use_stdin_without_content_in_argv_or_files(fake_workers, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    secret = "synthetic query with spaces, quotes ' \" and café"
    with webbatch.WebBatch([request(secret)]) as batch:
        result = dict(batch)[0]
    assert json.loads(result)["value"] == secret
    _, args, kwargs = fake_workers[0]
    assert all(secret not in arg for arg in args[0])
    assert kwargs["stdin"] is subprocess.PIPE and kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.DEVNULL and kwargs["start_new_session"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mode,message", [("bad", "invalid result"), ("large", "output exceeded"), ("exit", "status 7")])
def test_bad_worker_results_are_bounded_errors_and_do_not_cancel_siblings(fake_workers, mode, message):
    with webbatch.WebBatch([request(mode=mode), request("ok")]) as batch:
        results = dict(batch)
    assert message in results[0]
    assert json.loads(results[1])["value"] == "ok"


def test_timeout_kills_worker_and_returns_error(fake_workers):
    started = time.monotonic()
    with webbatch.WebBatch([request(delay=30)], timeout=0.15) as batch:
        results = dict(batch)
    assert "timed out" in results[0]
    assert time.monotonic() - started < 3


def test_interrupt_preserves_finished_results_and_never_starts_queued_requests(fake_workers):
    batch = webbatch.WebBatch([request("done"), *[request(i, delay=30) for i in range(1, 7)]])
    with pytest.raises(KeyboardInterrupt), batch:
        iterator = iter(batch)
        index, result = next(iterator)
        assert index == 0 and json.loads(result)["value"] == "done"
        raise KeyboardInterrupt
    assert set(batch.results) == {0}
    assert batch.started == {0, 1, 2, 3}
    assert len(fake_workers) == 4


def test_interrupt_after_spawn_reaps_worker_before_ownership_registration(fake_workers, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(webbatch, "_Job", interrupt)
    batch = webbatch.WebBatch([request(delay=30), request("never launched")])
    with pytest.raises(KeyboardInterrupt), batch:
        list(batch)
    assert len(fake_workers) == 1
    process = fake_workers[0][0]
    assert process.poll() is not None and process.stdin.closed and process.stdout.closed
    assert not batch._jobs and batch._closed


@pytest.mark.parametrize("during_close", [False, True])
def test_interrupt_during_stop_still_reaps_all_workers(fake_workers, monkeypatch, during_close):
    batch = webbatch.WebBatch([request(delay=30), request(delay=30)])
    batch._fill()  # workers wait on stdin; no synthetic request has been sent
    real_killpg = webbatch.os.killpg
    interrupted = False

    def killpg(pid, sig):
        nonlocal interrupted
        assert any(job.process.pid == pid for job in batch._jobs.values())
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        real_killpg(pid, sig)

    monkeypatch.setattr(webbatch.os, "killpg", killpg)
    with pytest.raises(KeyboardInterrupt), batch:
        if during_close:
            batch.close()
        else:
            batch._stop(0)
    assert interrupted and batch._closed and not batch._jobs
    assert len(fake_workers) == 2
    for process, _, _ in fake_workers:
        assert process.poll() is not None and process.stdin.closed and process.stdout.closed
    batch.close()  # cleanup remains idempotent after propagating the interrupt


def test_close_keeps_fully_received_result_just_before_process_exit(fake_workers):
    batch = webbatch.WebBatch([request("done", mode="linger"), request("unfinished", delay=30)])
    batch._fill()
    for job in batch._jobs.values():
        os.write(job.process.stdin.fileno(), job.pending)
        job.pending = job.pending[:0]
        batch._unregister_close(job.process.stdin)
    deadline = time.monotonic() + 3
    while not batch._jobs[0].output and time.monotonic() < deadline:
        batch._read(0)
        time.sleep(0.01)
    batch.close()
    assert json.loads(batch.results[0])["value"] == "done"
    assert 1 not in batch.results
    batch.close()  # idempotent


def test_final_bytes_written_between_nonblocking_read_and_exit_are_not_lost(monkeypatch):
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    stdout = os.fdopen(read_fd, "rb", buffering=0)
    batch = webbatch.WebBatch([])

    def poll():
        # The first read found no bytes available. The worker now writes the
        # rest of its result and exits before the parent observes its status.
        os.write(write_fd, b'plete"')
        os.close(write_fd)
        return 0

    process = SimpleNamespace(stdout=stdout, poll=poll)
    batch._jobs[0] = webbatch._Job(process, memoryview(b""), time.monotonic() + 10, bytearray(b'"com'))

    def stop(index):
        batch._jobs.pop(index).process.stdout.close()

    monkeypatch.setattr(batch, "_stop", stop)
    try:
        batch._collect()
        assert batch.results == {0: "complete"}
    finally:
        batch.close()
        stdout.close()


def test_invalid_requests_do_not_launch_workers_and_valid_siblings_run(fake_workers):
    requests = [
        {"name": "run_shell", "arguments": {"command": "bad"}},
        {"name": "web_search", "arguments": "not an object"},
        request("x" * (webbatch.MAX_REQUEST_BYTES + 1)),
        request("ok"),
    ]
    with webbatch.WebBatch(requests) as batch:
        results = dict(batch)
    assert all(results[index].startswith("error:") for index in range(3))
    assert batch.started == {3} and len(fake_workers) == 1
    assert json.loads(results[3])["value"] == "ok"


def test_spawn_failure_becomes_a_result_without_losing_other_requests(monkeypatch):
    monkeypatch.setattr(webbatch.subprocess, "Popen", lambda *a, **kw: (_ for _ in ()).throw(OSError("missing interpreter")))
    with webbatch.WebBatch([request(1), request(2)]) as batch:
        assert len(dict(batch)) == 2
    assert not batch.started and all("missing interpreter" in result for result in batch.results.values())


@pytest.mark.parametrize("name,arguments", [
    ("fetch_url", {"url": "https://example.test"}),
    ("web_search", {"query": "test", "max_results": 2}),
])
def test_worker_dispatches_only_the_selected_builtin_in_memory(monkeypatch, name, arguments):
    received = []

    def execute(**kwargs):
        received.append(kwargs)
        print("library diagnostic")
        return "result café"

    monkeypatch.setattr(tools, "fetch_url", execute)
    monkeypatch.setattr(tools, "web_search", execute)
    payload = json.dumps({"name": name, "arguments": arguments, "searx": "http://configured-search"}).encode()
    monkeypatch.setattr(web_worker.sys, "stdin", io.TextIOWrapper(io.BytesIO(payload)))
    stdout, stderr = io.StringIO(), io.StringIO()
    monkeypatch.setattr(web_worker.sys, "stdout", stdout)
    monkeypatch.setattr(web_worker.sys, "stderr", stderr)
    web_worker.main()
    assert json.loads(stdout.getvalue()) == "result café"
    assert "library diagnostic" in stderr.getvalue()
    expected = {**arguments, **({"searx": "http://configured-search"} if name == "web_search" else {})}
    assert received == [expected]


def test_worker_returns_errors_without_tracebacks(monkeypatch):
    monkeypatch.setattr(web_worker.sys, "stdin", io.TextIOWrapper(io.BytesIO(b'{"name":"run_shell","arguments":{}}')))
    output = io.StringIO()
    monkeypatch.setattr(web_worker.sys, "stdout", output)
    web_worker.main()
    assert json.loads(output.getvalue()) == "error: unknown web tool"


def test_real_worker_ignores_workspace_and_pythonpath_modules(tmp_path, monkeypatch):
    (tmp_path / "lethe.py").write_text("raise RuntimeError('UNTRUSTED WORKSPACE IMPORTED')\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    result = subprocess.run(
        webbatch._worker_argv(), cwd=tmp_path,
        input=json.dumps({"name": "fetch_url", "arguments": {"url": "file:///not-opened"}}),
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == "error: refusing non-http(s) URL: file:///not-opened"
    assert "UNTRUSTED" not in result.stderr


def test_empty_or_preclosed_batch_starts_nothing(fake_workers):
    with webbatch.WebBatch([]) as batch:
        assert list(batch) == []
    batch = webbatch.WebBatch([request()])
    batch.close()
    assert list(batch) == [] and not fake_workers
