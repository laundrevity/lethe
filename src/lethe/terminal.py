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
