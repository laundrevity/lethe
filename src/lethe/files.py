"""Opt-in file tools that ride along with the shell tool: read, write, exact-string edit.

read_file exists in both shell modes (confined to the workspace when read-only);
write_file and edit_file exist only in full-access mode. Every call asks y/n, and
writes show a diff first. Content goes only into the conversation and the file the
user approved: no backups, temp files, or edit history.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

MAX_FILE_BYTES = 2_000_000
MAX_READ_CHARS = 16_000  # same budget as other tool results
_NUMBERED = re.compile(r"^\s*\d+\t")


def _lines(text: str) -> list[str]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _diff(old: str, new: str) -> str:
    out = []
    for line in difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True), n=3):
        if line.startswith(("---", "+++")):
            continue
        out.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(out).rstrip("\n")


def _near_miss(text: str, old: str) -> tuple[int, int] | None:
    """Line span (0-based, exclusive end) matching `old` apart from indentation/trailing whitespace."""
    want = [ln.strip() for ln in old.strip("\n").split("\n")]
    have = text.split("\n")
    if not any(want):
        return None
    for i in range(len(have) - len(want) + 1):
        if all(have[i + j].strip() == w for j, w in enumerate(want)):
            return i, i + len(want)
    return None


def apply_edit(text: str, old: str, new: str, replace_all: bool = False) -> tuple[str, int]:
    """Exact-string replacement. Errors are written for the model: each says how to fix the call."""
    if old == "":
        raise ValueError("old_string must not be empty; use write_file to create a file or replace all of it")
    if old == new:
        raise ValueError("old_string and new_string are identical; nothing to change")
    if old not in text and "\r\n" in text and "\r\n" not in old:
        # the file uses CRLF line endings; models always send LF
        old, new = old.replace("\n", "\r\n"), new.replace("\r\n", "\n").replace("\n", "\r\n")
    count = text.count(old)
    if count == 0:
        nonblank = [ln for ln in old.split("\n") if ln.strip()]
        if nonblank and all(_NUMBERED.match(ln) for ln in nonblank):
            raise ValueError("old_string not found: it contains read_file's line-number prefixes; "
                             "remove the leading 'N<tab>' from every line")
        if span := _near_miss(text, old):
            actual = "\n".join(text.split("\n")[span[0]:span[1]])
            raise ValueError(f"old_string not found exactly, but lines {span[0] + 1}-{span[1]} match apart from "
                             f"whitespace. Retry with the exact text:\n{actual}")
        raise ValueError("old_string not found in the file; read_file the relevant lines and copy them exactly")
    if count > 1 and not replace_all:
        at, pos = [], text.find(old)
        while pos != -1 and len(at) < 10:
            at.append(str(text.count("\n", 0, pos) + 1))
            pos = text.find(old, pos + len(old))
        raise ValueError(f"old_string matches {count} places (starting at lines {', '.join(at)}); include more "
                         "surrounding lines to make it unique, or set replace_all to true")
    return text.replace(old, new), count


@dataclass(frozen=True)
class FileTools:
    root: Path  # already resolved by the shell runner
    confined: bool = True  # read-only mode: paths must stay inside root

    def _resolve(self, path: str) -> Path:
        if not isinstance(path, str) or not path.strip() or "\0" in path:
            raise ValueError("path must be a nonempty string without NUL bytes")
        p = (self.root / Path(path).expanduser()).resolve()
        if self.confined:
            if self.root.resolve(strict=True) != self.root:
                raise ValueError("shell workspace changed; restart with the intended directory")
            if not p.is_relative_to(self.root):
                raise ValueError("path must be within the shell workspace")
        return p

    def _show(self, p: Path) -> str:
        return str(p.relative_to(self.root)) if p.is_relative_to(self.root) else str(p)

    def _read_text(self, p: Path) -> str:
        if p.is_dir():
            raise ValueError(f"{self._show(p)} is a directory; list it with run_shell")
        if not p.is_file():
            raise ValueError(f"no such file: {self._show(p)}")
        if p.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{self._show(p)} is larger than {MAX_FILE_BYTES:,} bytes; inspect it with run_shell")
        try:
            return p.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"{self._show(p)} is not UTF-8 text") from None

    def read_file(self, path: str, offset: int = 1, limit: int | None = None) -> str:
        p = self._resolve(path)
        lines = _lines(self._read_text(p))
        if not lines:
            return "[empty file]"
        if offset > len(lines):
            return f"[offset {offset} is past the end: the file has {len(lines)} lines]"
        end = len(lines) if limit is None else min(len(lines), offset - 1 + limit)
        out, used, n = [], 0, offset
        while n <= end:
            row = f"{n}\t{lines[n - 1].rstrip(chr(13))}"
            if used + len(row) + 1 > MAX_READ_CHARS:
                if not out:  # one enormous line (minified/generated): show its head rather than nothing
                    out.append(row[:MAX_READ_CHARS] + " […line truncated]")
                    n += 1
                break
            out.append(row)
            used += len(row) + 1
            n += 1
        if n <= len(lines):
            out.append(f"[lines {offset}-{n - 1} of {len(lines)}; call read_file with offset={n} to continue]")
        return "\n".join(out)

    def _planned_write(self, path: str, content: str) -> tuple[Path, str | None]:
        p = self._resolve(path)
        if p.is_dir():
            raise ValueError(f"{self._show(p)} is a directory")
        if not p.exists():
            return p, None
        try:
            old = self._read_text(p)
        except ValueError:
            return p, ""  # binary or oversized: shown as a full replacement
        if old == content:
            raise ValueError(f"{self._show(p)} already has exactly this content")
        return p, old

    def preview_write(self, path: str, content: str) -> str:
        p, old = self._planned_write(path, content)
        verb = "create" if old is None else "overwrite"
        return f"{verb} {p} ({len(_lines(content))} lines)\n{_diff(old or '', content)}"

    def write_file(self, path: str, content: str) -> str:
        p, old = self._planned_write(path, content)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            f.write(content)
        return f"{'created' if old is None else 'overwrote'} {self._show(p)} ({len(_lines(content))} lines)"

    def _planned_edit(self, path: str, old_string: str, new_string: str, replace_all: bool) -> tuple[Path, str, str, int]:
        p = self._resolve(path)
        text = self._read_text(p)
        updated, count = apply_edit(text, old_string, new_string, replace_all)
        return p, text, updated, count

    def preview_edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        p, text, updated, count = self._planned_edit(path, old_string, new_string, replace_all)
        return f"edit {p} ({count} replacement{'s' if count != 1 else ''})\n{_diff(text, updated)}"

    def edit_file(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        p, _, updated, count = self._planned_edit(path, old_string, new_string, replace_all)
        with p.open("w", encoding="utf-8", newline="") as f:
            f.write(updated)
        return f"edited {self._show(p)}: {count} replacement{'s' if count != 1 else ''}"
