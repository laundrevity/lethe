"""Bounded web workers, with requests and results carried only through pipes.

All processes belong to the current batch. Closing it stops their process groups,
including curl children, so Ctrl-C cannot leave requests running in the background.
Approval and terminal rendering remain in the caller's main thread.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator

MAX_WORKERS = 4
WORKER_TIMEOUT = 60.0
MAX_REQUEST_BYTES = 128_000
MAX_RESULT_BYTES = 128_000


def _worker_argv() -> list[str]:
    # Import only the installed Lethe package, never a lethe.py in the user's
    # workspace or a module selected by PYTHONPATH. Network env vars still work.
    return [sys.executable, "-I", "-B", "-m", "lethe.web_worker"]


@dataclass
class _Job:
    process: subprocess.Popen
    pending: memoryview
    deadline: float
    output: bytearray = field(default_factory=bytearray)


class WebBatch:
    """Yield ``(request index, result)`` as approved web calls complete.

    ``results`` also contains completions collected during close/interrupt, even
    if the iterator did not yield them yet. ``started`` distinguishes requests
    that may have reached the network from those never launched. The caller can
    use these to preserve source-order history after an interrupt.
    """

    def __init__(self, requests: list[dict], searx: str | None = None, *, timeout: float = WORKER_TIMEOUT):
        if timeout <= 0:
            raise ValueError("web worker timeout must be positive")
        self.requests = requests
        self.searx = searx
        self.timeout = timeout
        self.results: dict[int, str] = {}
        self.started: set[int] = set()
        self._jobs: dict[int, _Job] = {}
        self._ready: list[tuple[int, str]] = []
        self._selector = selectors.DefaultSelector()
        self._next = 0
        self._closed = False
        self._closing = False
        self._iterated = False

    def __enter__(self) -> WebBatch:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _record(self, index: int, result: str) -> None:
        self.results[index] = result
        self._ready.append((index, result))

    def _unregister_close(self, stream) -> None:
        if stream.closed:
            return
        try:
            self._selector.unregister(stream)
        except KeyError:
            pass
        stream.close()

    def _reap(self, process: subprocess.Popen) -> bool:
        """Finish cleanup before reporting a Ctrl-C received during cleanup."""
        interrupted = False
        while True:
            try:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._unregister_close(process.stdin)
                self._unregister_close(process.stdout)
                process.wait()
                return interrupted
            except KeyboardInterrupt:
                interrupted = True

    def _stop(self, index: int) -> None:
        job = self._jobs[index]
        interrupted = self._reap(job.process)
        del self._jobs[index]
        if interrupted:
            raise KeyboardInterrupt

    def _finish(self, index: int, result: str) -> None:
        self._record(index, result)
        self._stop(index)

    def _launch(self, index: int) -> None:
        request = self.requests[index]
        try:
            if not isinstance(request, dict) or request.get("name") not in {"web_search", "fetch_url"} or not isinstance(request.get("arguments"), dict):
                raise ValueError("only web_search and fetch_url requests are supported")
            payload = json.dumps({**request, "searx": self.searx}, ensure_ascii=False).encode("utf-8")
            if len(payload) > MAX_REQUEST_BYTES:
                raise ValueError(f"web request exceeds {MAX_REQUEST_BYTES:,} bytes")
            process = subprocess.Popen(
                _worker_argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                start_new_session=True, close_fds=True,
            )
        except (OSError, ValueError, TypeError) as error:
            self._record(index, f"error: {error}")
            return
        try:
            self.started.add(index)
            job = _Job(process, memoryview(payload), time.monotonic() + self.timeout)
            self._jobs[index] = job
            for stream, event, kind in ((process.stdin, selectors.EVENT_WRITE, "stdin"),
                                        (process.stdout, selectors.EVENT_READ, "stdout")):
                os.set_blocking(stream.fileno(), False)
                self._selector.register(stream, event, (index, kind))
        except BaseException:
            # An interrupt can arrive before _Job has taken ownership, too.
            interrupted = self._reap(process)
            self._jobs.pop(index, None)
            if interrupted:
                raise KeyboardInterrupt
            raise

    def _fill(self) -> None:
        while not self._closing and self._next < len(self.requests) and len(self._jobs) < MAX_WORKERS:
            index = self._next
            self._next += 1
            self._launch(index)

    def _read(self, index: int) -> None:
        job = self._jobs[index]
        if job.process.stdout.closed:
            return
        while True:
            try:
                chunk = os.read(job.process.stdout.fileno(), 16_384)
            except BlockingIOError:
                return
            if not chunk:
                self._unregister_close(job.process.stdout)
                return
            if len(job.output) + len(chunk) > MAX_RESULT_BYTES:
                self._finish(index, f"error: web worker output exceeded {MAX_RESULT_BYTES:,} bytes")
                return
            job.output.extend(chunk)

    @staticmethod
    def _decode(output: bytearray) -> str | None:
        try:
            result = json.loads(output)
        except (ValueError, UnicodeError):
            return None
        return result if isinstance(result, str) else None

    def _collect(self, *, closing: bool = False) -> None:
        for index, job in list(self._jobs.items()):
            self._read(index)
            if index not in self._jobs:
                continue  # output overflow already completed and stopped it
            returncode = job.process.poll()
            if returncode is not None:
                # The worker may write its last bytes and exit between the
                # first nonblocking read and poll(). Drain again after exit.
                self._read(index)
                if index not in self._jobs:
                    continue
            result = self._decode(job.output)
            if closing and result is not None:
                # The worker emits exactly one result at the end. Keep a fully
                # received result even if interrupted just before process exit.
                self._finish(index, result)
            elif returncode is not None:
                if returncode != 0:
                    result = f"error: web worker exited with status {returncode}"
                elif result is None:
                    result = "error: web worker returned an invalid result"
                self._finish(index, result)
            elif time.monotonic() >= job.deadline:
                self._finish(index, f"error: web request timed out after {self.timeout:g} seconds")

    def __iter__(self) -> Iterator[tuple[int, str]]:
        if self._iterated:
            raise RuntimeError("a web batch can only be iterated once")
        self._iterated = True
        try:
            if self._closed:
                return
            self._fill()
            while self._jobs or self._ready:
                for key, _ in self._selector.select(0.05 if self._jobs else 0):
                    index, kind = key.data
                    if index not in self._jobs:
                        continue
                    job = self._jobs[index]
                    if kind == "stdin":
                        try:
                            job.pending = job.pending[os.write(key.fd, job.pending):]
                        except BrokenPipeError:
                            job.pending = job.pending[:0]
                        if not job.pending:
                            self._unregister_close(job.process.stdin)
                    else:
                        self._read(index)
                self._collect()
                ready, self._ready = self._ready, []
                yield from ready
                self._fill()
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        interrupted = False
        try:
            self._collect(closing=True)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            while self._jobs:
                try:
                    self._stop(next(iter(self._jobs)))
                except KeyboardInterrupt:
                    interrupted = True
            self._selector.close()
            self._closed = True
        if interrupted:
            raise KeyboardInterrupt
