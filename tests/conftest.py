"""Shared fixtures for glyphflip tests.

Heavy things (model downloads, real OCR runs) are kept out of the default
suite. Synthetic images are PIL-rendered so we get deterministic text +
bounding boxes without any network or model weight dependencies.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Pick a usable TTF font across systems; fall back to bitmap default.

    On Arch, DejaVuSans ships with `ttf-dejavu` (usually present).
    macOS has it bundled with matplotlib or at /Library/Fonts on some setups;
    Windows ships Arial. We try a handful and degrade gracefully.
    """
    candidates = [
        "/usr/share/fonts/TTF/DejaVuSans.ttf",      # Arch
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian/Ubuntu
        "/System/Library/Fonts/Helvetica.ttc",       # macOS
        "C:/Windows/Fonts/arial.ttf",                # Windows
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_text_image(
    lines: list[str],
    *,
    width: int = 640,
    height: int | None = None,
    font_size: int = 32,
    padding: int = 20,
    fmt: str = "PNG",
) -> bytes:
    """Render simple black-on-white text lines into an image buffer.

    Returns raw bytes in the requested format so tests can exercise both
    Path-based and bytes-based backend input paths.
    """
    font = _load_font(font_size)
    line_height = font_size + 10
    height = height or (padding * 2 + line_height * len(lines))

    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill="black", font=font)
        y += line_height

    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def simple_english_png() -> bytes:
    """Single-line known text, ideal for adapter smoke tests."""
    return render_text_image(["Hello glyphflip"])


@pytest.fixture
def multiline_english_png() -> bytes:
    """Three lines with distinct vertical offsets, for ordering tests."""
    return render_text_image(["first line", "second line", "third line"])


@pytest.fixture
def tmp_png(tmp_path: Path, simple_english_png: bytes) -> Path:
    """Same image as a real file on disk, exercising the Path input path."""
    p = tmp_path / "input.png"
    p.write_bytes(simple_english_png)
    return p

