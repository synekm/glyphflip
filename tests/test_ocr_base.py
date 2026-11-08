from glyphflip.ocr.base import (
    OcrOptions, OcrResult, TextBox, clamp_options, UnsupportedLanguageError,
)
import pytest

def test_reading_order():
    boxes = (
        TextBox("world", (100.0, 60.0, 200.0, 80.0)),
        TextBox("hello", (10.0, 10.0, 90.0, 30.0)),
    )
    assert OcrResult(boxes).raw_text == "hello\nworld"

def test_clamp_rejects_unknown_lang():
    with pytest.raises(UnsupportedLanguageError):
        clamp_options(OcrOptions(language="xx"))

def test_clamp_clamps_confidence():
    o = clamp_options(OcrOptions(min_confidence=2.0))
    assert o.min_confidence == 1.0
