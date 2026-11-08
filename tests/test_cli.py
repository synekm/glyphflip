"""CLI tests via typer's CliRunner — no models, pipeline faked.

Verifies argument plumbing, error exits, and output formatting only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from glyphflip.cli import app
from glyphflip.ocr.base import OcrResult, TextBox
from glyphflip.pipeline import GlyphFlipResult
from glyphflip.translate.base import TranslationResult

runner = CliRunner()


class FakePipe:
    """Stands in for GlyphFlipPipeline.process()."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def process(self, image, target_language, *, source_language=None,
                ocr_options=None):
        self.calls.append((target_language, source_language))
        boxes = (TextBox("Ahoj", (0.0, 0.0, 50.0, 20.0)),)
        return GlyphFlipResult(
            ocr=OcrResult(boxes=boxes),
            translation=TranslationResult(("Hello",), "cs", "en"),
            ordered_boxes=boxes,
        )


@pytest.fixture
def fake_pipe(monkeypatch):
    pipe = FakePipe()
    monkeypatch.setattr("glyphflip.pipeline.GlyphFlipPipeline", lambda *a, **k: pipe)
    # CLI factories construct engines lazily; neuter them so no models load
    monkeypatch.setattr(
        "glyphflip.ocr.rapidocr_backend.RapidocrBackend",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr(
        "glyphflip.translate.opus_mt.OpusMtTranslator",
        lambda *a, **k: object(),
    )
    return pipe


def _make_image(tmp_path: Path) -> Path:
    import io

    from PIL import Image

    p = tmp_path / "in.png"
    Image.new("RGB", (60, 30), "white").save(p, format="PNG")
    return p


def test_translate_prints_translated_text(fake_pipe, tmp_path):
    img = _make_image(tmp_path)
    res = runner.invoke(app, ["translate", str(img), "--to", "en", "--from", "cs"])
    assert res.exit_code == 0, res.output
    assert "Hello" in res.output
    # arguments flowed through correctly
    assert fake_pipe.calls[0] == ("en", "cs")


def test_translate_missing_file_exits_nonzero(tmp_path):
    res = runner.invoke(app, ["translate", str(tmp_path / "ghost.png"), "--to", "en"])
    assert res.exit_code == 1
    assert "no such file" in res.output.lower()


def test_translate_bad_target_language_exits_2(tmp_path, fake_pipe):
    img = _make_image(tmp_path)
    res = runner.invoke(app, ["translate", str(img), "--to", "xx"])
    assert res.exit_code == 2
    assert "unsupported target" in res.output.lower()


def test_version_flag():
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0
    assert "0.1.0" in res.output


def test_help_requires_no_models():
    """--help must not import torch or onnxruntime at all."""
    import sys

    before = set(sys.modules)
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    loaded = set(sys.modules) - before
    assert not {"torch", "onnxruntime"} & loaded

