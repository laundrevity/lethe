"""Turn local files into prompt content. Reads only; writes nothing."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

TEXT_SUFFIXES = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".csv", ".tsv", ".html", ".htm", ".xml", ".css", ".sh", ".zsh", ".bash", ".c", ".h", ".cpp", ".hpp", ".cc", ".rs",
    ".go", ".java", ".kt", ".swift", ".rb", ".php", ".sql", ".tex", ".log", ".diff", ".patch", ".m", ".mm", ".r",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mpeg", ".mpg"}


class UnsupportedFile(Exception):
    pass


def read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, 1):
            t = page.extract_text() or ""
            if t.strip():
                pages.append(f"[page {i}]\n{t}")
        return "\n\n".join(pages)
    if suffix in TEXT_SUFFIXES or suffix == "":
        return path.read_text(errors="replace")
    # sniff: if it decodes as utf-8 without NULs, treat as text
    raw = path.read_bytes()[:65536]
    if b"\0" not in raw:
        try:
            raw.decode("utf-8")
            return path.read_text(errors="replace")
        except UnicodeDecodeError:
            pass
    raise UnsupportedFile(f"{path.name}: not a text or PDF file")


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def file_block(path: Path) -> str:
    text = read_text(path)
    return f'<file name="{path.name}">\n{text}\n</file>'


def image_data_url(path: Path) -> str:
    if path.suffix.lower() == ".webp":
        # llama.cpp's WebP fallback uses FFmpeg's disk-backed cache protocol.
        # Decode the selected local file directly instead, keeping PNG in RAM.
        from .video import image_png

        return "data:image/png;base64," + base64.b64encode(image_png(path)).decode()
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def approx_tokens(text: str) -> int:
    """Rough token estimate for the UI (~4 chars/token for English, 1 per CJK char)."""
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    return int((len(text) - cjk) / 4 + cjk)
