"""Terminal-side presentation: a status line pinned under the streaming reply, and attention notifications.

Everything here goes to the terminal and nowhere else. Notification text is a fixed phrase, never
conversation content: macOS keeps notification text in Notification Center until it is dismissed.
"""

from __future__ import annotations

import os
import sys
import time
import unicodedata
from typing import TextIO

APPLE_TERMINAL = "Apple_Terminal"

_DISPLAY_ESCAPES = {
    code: f"\\x{code:02x}" for code in [*range(32), *range(127, 160)] if code not in (9, 10)
}
_DISPLAY_ESCAPES.update({code: f"\\u{code:04x}" for code in [*range(0x202A, 0x202F), *range(0x2066, 0x206A)]})


def terminal_safe(text: str) -> str:
    """Escape terminal controls per chunk, including split sequences, without changing model context."""
    return text.translate(_DISPLAY_ESCAPES)


class _CodeArgumentsPreview:
    """Decode JSON string escapes for display, without parsing or executing a call.

    Keep the surrounding punctuation so every field remains visible. Only the
    current escape (at most six characters) and a UTF-16 high surrogate need to
    wait for another chunk. This never changes the arguments stored in history.
    """

    _ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}

    def __init__(self):
        self.in_string = False
        self.escape = ""
        self.high_surrogate: int | None = None

    def _character(self, ch: str) -> str:
        point = ord(ch)
        prefix = ""
        if self.high_surrogate is not None:
            high, self.high_surrogate = self.high_surrogate, None
            if 0xDC00 <= point <= 0xDFFF:
                return chr(0x10000 + ((high - 0xD800) << 10) + point - 0xDC00)
            prefix = f"\\u{high:04x}"
        if 0xD800 <= point <= 0xDBFF:
            self.high_surrogate = point
            return prefix
        return prefix + (f"\\u{point:04x}" if 0xDC00 <= point <= 0xDFFF else ch)

    def pending(self) -> str:
        """Show incomplete escapes literally when a stream ends or is interrupted."""
        tail = f"\\u{self.high_surrogate:04x}" if self.high_surrogate is not None else ""
        self.high_surrogate = None
        tail += self.escape
        self.escape = ""
        return tail

    def feed(self, text: str) -> str:
        out = []
        for ch in text:
            if self.escape:
                self.escape += ch
                if len(self.escape) == 2:
                    if ch == "u":
                        continue
                    out.append(self._character(self._ESCAPES[ch]) if ch in self._ESCAPES else self.pending())
                    self.escape = ""
                elif ch not in "0123456789abcdefABCDEF":
                    out.append(self.pending())
                elif len(self.escape) == 6:
                    out.append(self._character(chr(int(self.escape[2:], 16))))
                    self.escape = ""
                continue
            if self.in_string:
                if ch == "\\":
                    self.escape = ch
                elif ch == '"':
                    out.append(self.pending() + ch)
                    self.in_string = False
                else:
                    out.append(self._character(ch))
            else:
                if ch == '"':
                    self.in_string = True
                out.append(ch + "\n" if ch == "," else ch)
        return "".join(out)


class ToolCallStream:
    """Append tool names and arguments as they arrive, labeled by call index.

    Python, Bash and SSH code get decoded previews; other tools keep raw JSON. This is
    presentation only. A matching, contiguous preview can be reused for approval;
    interleaved or changed calls need a complete final rendering instead.
    """

    def __init__(self, status: StatusLine, *, color: bool | None = None):
        self.status = status
        self.out = status.out
        if color is None:
            color = "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
        self.color = bool(color and _is_tty(self.out))
        self.names: dict[int, str] = {}
        self.ids: dict[int, str] = {}
        self.arguments: dict[int, list[str]] = {}
        self.segments: dict[int, int] = {}
        self.unusable: set[int] = set()
        self.code_previews: dict[int, _CodeArgumentsPreview] = {}
        self.index: int | None = None
        self.in_arguments = False

    def __enter__(self) -> ToolCallStream:
        return self

    def __exit__(self, *exc) -> None:
        for index, preview in self.code_previews.items():
            if tail := preview.pending():
                self._select(index)
                if not self.in_arguments:
                    self._write("\n")
                    self.in_arguments = True
                self._write(tail)
        self.finish()

    def _write(self, text: str) -> None:
        text = terminal_safe(text)
        # Color each fragment independently: status redraws reset attributes.
        # Only visible text participates in right-margin/wrap accounting.
        self.out.write(f"\x1b[36m{text}\x1b[0m" if self.color and text else text)
        self.status.wrote(text)

    def _select(self, index: int) -> None:
        if index != self.index:
            self.finish()
            if self.status.col:
                self._write("\n")
            self.index = index
            self.segments[index] = self.segments.get(index, 0) + 1
            self._write(f"⚙ tool call {index + 1}: {self.names[index]}")

    def feed(self, fragments: list[dict]) -> None:
        for fragment in fragments:
            index = fragment.get("index", 0)
            function = fragment.get("function") or {}
            name, arguments = function.get("name") or "", function.get("arguments") or ""
            self.ids[index] = fragment.get("id") or self.ids.get(index, f"call_{index}")
            if name and self.arguments.get(index):
                # A late name may have selected a different decoder midway.
                self.unusable.add(index)
            self.names[index] = self.names.get(index, "") + name
            if name and self.in_arguments and index == self.index:
                self.finish()
            if index != self.index:
                self._select(index)
            elif name:
                self._write(name)
            if arguments:
                self.arguments.setdefault(index, []).append(arguments)
                if not self.in_arguments:
                    self._write("\n")
                    self.in_arguments = True
                if self.names[index] in {"run_ssh", "run_python", "run_bash"}:
                    preview = self.code_previews.setdefault(index, _CodeArgumentsPreview())
                    self._write(preview.feed(arguments))
                else:
                    self._write(arguments)
        self.out.flush()

    def shown_call(self, call: dict) -> int | None:
        """Return the 1-based label of an exact, uninterrupted displayed call.

        This is only a display receipt. The caller must independently establish
        stream completion and validate the arguments before asking for approval.
        """
        function = call.get("function") or {}
        call_id = call.get("id") or f"call_{call.get('index', 0)}"
        matches = [index for index, shown_id in self.ids.items() if shown_id == call_id]
        if len(matches) != 1:
            return None
        index = matches[0]
        if index in self.unusable or self.segments.get(index) != 1 or index not in self.arguments:
            return None
        if self.names[index] != function.get("name") or "".join(self.arguments[index]) != function.get("arguments"):
            return None
        preview = self.code_previews.get(index)
        if preview is not None and (preview.escape or preview.high_surrogate is not None):
            return None  # Some source has not reached the terminal yet.
        return index + 1

    def finish(self) -> None:
        if self.index is not None:
            if self.status.col:
                self._write("\n")
            self.index, self.in_arguments = None, False
            self.out.flush()


def _is_tty(out: TextIO) -> bool:
    try:
        return bool(out.isatty())
    except (AttributeError, ValueError):
        return False


def notify(message: str, out: TextIO | None = None) -> None:
    """Ask for attention: the terminal bell (a tab badge or Dock bounce in Apple Terminal, which has
    nothing else) plus a desktop notification where the terminal supports one (OSC 9: iTerm2, Ghostty,
    WezTerm; OSC 99: kitty). `message` must be a fixed phrase, never conversation content."""
    out = sys.stdout if out is None else out
    if not _is_tty(out):
        return
    sequence = "\a"
    if os.environ.get("KITTY_WINDOW_ID"):
        sequence += f"\x1b]99;i=1:d=0;{message}\x1b\\"
    elif os.environ.get("TERM_PROGRAM") != APPLE_TERMINAL:
        sequence += f"\x1b]9;{message}\x1b\\"
    out.write(sequence)
    out.flush()


def _cells(ch: str) -> int:
    """Screen cells a character takes: 0 for combining/format marks, 2 for East Asian wide, else 1."""
    if unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Me", "Cf"):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


class StatusLine:
    """One line of status pinned to the bottom row while a reply streams above it.

    The rows above the last one become the scroll region (DECSTBM), so streamed text scrolls and lands
    in the scrollback as usual — verified in Apple Terminal; iTerm2 and xterm keep lines scrolled out
    of a region anchored at the top too. Set it up at the start of a line. Inactive, and free, when
    stdout is not a terminal or the window is tiny.

    Redraws save and restore the cursor. Apple Terminal does not keep the pending line wrap with it,
    so the caller reports what it writes (`wrote`) and no redraw happens while the cursor sits at the
    right margin; the next chunk of output moves it on and the redraw follows.
    """

    MIN_ROWS, MIN_COLS = 6, 20
    INTERVAL = 0.2  # seconds between redraws while text streams

    def __init__(self, out: TextIO | None = None):
        self.out = sys.stdout if out is None else out
        self.active = False
        self.rows = self.cols = 0
        self.col = 0  # cells written since the caller's last newline
        self.text = ""
        self._drawn = 0.0

    def __enter__(self) -> "StatusLine":
        if _is_tty(self.out):
            rows, cols = self._size()
            if rows >= self.MIN_ROWS and cols >= self.MIN_COLS:
                self.rows, self.cols = rows, cols
                self._reserve()
                self.active = True
        return self

    def __exit__(self, *exc) -> None:
        if self.active:
            self.active = False
            self._release(clear=True)

    def _size(self) -> tuple[int, int]:
        try:
            size = os.get_terminal_size(self.out.fileno())
        except (AttributeError, OSError, ValueError):
            return 0, 0
        return size.lines, size.columns

    def _reserve(self) -> None:
        # Index (down one row, scrolling if on the last row) then cursor-up puts the cursor back where it
        # was, column included, with the last row now free. DECSTBM homes the cursor, hence the save/restore.
        self.out.write(f"\x1bD\x1b[A\x1b7\x1b[1;{self.rows - 1}r\x1b8")
        self.out.flush()

    def _release(self, *, clear: bool) -> None:
        # Give the whole screen back; the region reset homes the cursor too, hence the save/restore.
        wipe = f"\x1b[{self.rows};1H\x1b[2K" if clear else ""
        self.out.write(f"\x1b7{wipe}\x1b[r\x1b8")
        self.out.flush()

    def wrote(self, text: str) -> None:
        """Account for visible text the caller wrote to the terminal (no escape sequences)."""
        for ch in text:
            if ch == "\n":
                self.col = 0
            elif ch == "\t":
                self.col += 8 - self.col % 8
            else:
                self.col += _cells(ch)

    @property
    def at_margin(self) -> bool:
        return self.col > 0 and self.cols > 0 and self.col % self.cols == 0

    def update(self, text: str, *, sgr: str = "\x1b[0m", force: bool = False) -> None:
        """Show `text` (throttled unless `force`); `sgr` re-applies the caller's text attributes afterwards,
        since not every terminal restores them with the cursor."""
        self.text = text
        if not self.active or self.at_margin:
            return
        now = time.monotonic()
        if not force and now - self._drawn < self.INTERVAL:
            return
        self._drawn = now
        rows, cols = self._size()
        if (rows, cols) != (self.rows, self.cols):  # the window was resized: move the reserved row
            self._release(clear=False)
            if rows < self.MIN_ROWS or cols < self.MIN_COLS:
                self.active = False
                return
            self.rows, self.cols = rows, cols
            self._reserve()
        self.out.write(f"\x1b7\x1b[{self.rows};1H\x1b[0;2m{text[: self.cols - 1]}\x1b[0m\x1b[K\x1b8{sgr}")
        self.out.flush()
