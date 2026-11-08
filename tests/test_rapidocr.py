"""Round-trip tests for the RapidOCR backend.

These run the real engine against PIL-rendered fixtures: slow on first
call (model load), but deterministic — synthetic images have known text.
"""

from __future__ import annotations

import pytest

from glyphflip.ocr.base import (
    ImageLoadError,
    OcrOptions,
    UnsupportedLanguageError,
)
from glyphflip.ocr.rapidocr_backend import RapidocrBackend

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def backend() -> RapidocrBackend:
    # module-scoped: engine load is expensive, share it across tests
    return RapidocrBackend()


def test_single_line_roundtrip(backend: RapidocrBackend, simple_english_png: bytes) -> None:
    result = backend.recognize(simple_english_png, OcrOptions())

    assert result.boxes, "expected at least one text box"
    joined = " ".join(b.text.lower() for b in result.boxes)
    assert "glyphflip" in joined

    # confidences came through as floats in [0, 1]
    for box in result.boxes:
        assert box.confidence is not None
        assert 0.0 <= box.confidence <= 1.0


def test_multiline_reading_order(
    backend: RapidocrBackend, multiline_english_png: bytes
) -> None:
    result = backend.recognize(multiline_english_png, OcrOptions())
    assert result.raw_text.lower().splitlines() == [
        "first line",
        "second line",
        "third line",
    ]


def test_bboxes_are_sane(backend: RapidocrBackend, simple_english_png: bytes) -> None:
    result = backend.recognize(simple_english_png, OcrOptions())
    for box in result.boxes:
        xmin, ymin, xmax, ymax = box.bbox
        assert 0.0 <= xmin < xmax
        assert 0.0 <= ymin < ymax


def test_path_input_matches_bytes_input(
    backend: RapidocrBackend, tmp_png: object, simple_english_png: bytes
) -> None:
    from glyphflip.ocr.base import OcrResult  # noqa: F401  (shape check aid)

    from_path = backend.recognize(tmp_png, OcrOptions())  # type: ignore[arg-type]
    from_bytes = backend.recognize(simple_english_png, OcrOptions())
    assert from_path.raw_text == from_bytes.raw_text


def test_min_confidence_filters_boxes(
    backend: RapidocrBackend, simple_english_png: bytes
) -> None:
    loose = backend.recognize(simple_english_png, OcrOptions(min_confidence=0.0))
    strict = backend.recognize(simple_english_png, OcrOptions(min_confidence=1.0))
    assert len(strict.boxes) <= len(loose.boxes)
    for box in strict.boxes:
        assert box.confidence >= 1.0


def test_unsupported_language_raises(backend: RapidocrBackend, simple_english_png: bytes) -> None:
    with pytest.raises(UnsupportedLanguageError):
        backend.recognize(simple_english_png, OcrOptions(language="ja"))


def test_garbage_bytes_raise_image_error(backend: RapidocrBackend) -> None:
    with pytest.raises(ImageLoadError):
        backend.recognize(b"\x00\x01\x02not-an-image", OcrOptions())


def test_missing_file_raises_image_error(backend: RapidocrBackend, tmp_path: object) -> None:
    from pathlib import Path

    with pytest.raises(ImageLoadError):
        backend.recognize(Path(tmp_path) / "nope.png", OcrOptions())  # type: ignore[arg-type]

