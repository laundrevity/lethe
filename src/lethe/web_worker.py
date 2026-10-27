"""One approved web operation. Protocol data uses stdin/stdout, never files."""

from __future__ import annotations

import contextlib
import json
import sys

from .webbatch import MAX_REQUEST_BYTES


def main() -> None:
    output = sys.stdout
    try:
        request = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(request) > MAX_REQUEST_BYTES:
            raise ValueError("web request is too large")
        request = json.loads(request)
        # Third-party diagnostics must not corrupt the protocol or reach disk.
        # The parent directs stderr to /dev/null.
        with contextlib.redirect_stdout(sys.stderr):
            from .tools import fetch_url, web_search

            arguments = request["arguments"]
            if request["name"] == "web_search":
                result = web_search(**arguments, searx=request.get("searx"))
            elif request["name"] == "fetch_url":
                result = fetch_url(**arguments)
            else:
                raise ValueError("unknown web tool")
    except Exception as error:
        result = f"error: {error}"[:16_000]
    output.write(json.dumps(result, ensure_ascii=False))
    output.flush()


if __name__ == "__main__":
    main()
