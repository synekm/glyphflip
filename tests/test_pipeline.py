"""Pipeline tests using fake OCR/translator backends.

No real models: we verify routing, ordering, alignment and error wrapping
against deterministic stubs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from glyphflip.ocr.base import (
    OcrBackend,
    OcrOptions,
    OcrResult,
    TextBox,
)
from glyphflip.pipeline import (
    GlyphFlipPipeline,
    PipelineError,
)
from glyphflip.translate.base import (
    TranslationRequest,
    TranslationResult,
    Translator,
    UnsupportedPairError,
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeOcr(OcrBackend):
    name = "fake-ocr"
    languages = frozenset({"en", "cs"})

    def __init__(self, boxes: tuple[TextBox, ...]) -> None:
        self._boxes = boxes

    def recognize(self, image: Path | bytes, options: OcrOptions) -> OcrResult:
        return OcrResult(boxes=self._boxes)

    def warm_up(self, language: str) -> None:
        pass


class FakeTranslator(Translator):
    name = "fake-translator"

    def __init__(
        self,
        served_pairs: frozenset[tuple[str, str]],
        *,
        fail: bool = False,
        upper: bool = False,
    ) -> None:
        self.pairs = served_pairs
        self._fail = fail
        self._upper = upper
        self.calls: list[TranslationRequest] = []

    def supports(self, source: str | None, target: str) -> bool:
        if source is None:
            return target in {t for _, t in self.pairs}
        return (source, target) in self.pairs

    def translate(self, request: TranslationRequest) -> TranslationResult:
        self.calls.append(request)
        if self._fail:
            raise RuntimeError("engine exploded")
        out = tuple(
            s.upper() if self._upper else f"<{s}>" for s in request.segments
        )
        return TranslationResult(out, request.source_language or "en",
                                 request.target_language)

    def warm_up(self, pair: tuple[str, str]) -> None:
        pass


BOXES = (
    TextBox("second", (100.0, 200.0, 200.0, 220.0)),
    TextBox("first", (10.0, 10.0, 90.0, 30.0)),
    TextBox("third", (50.0, 400.0, 150.0, 420.0)),
)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_process_routes_to_supporting_translator() -> None:
    local = FakeTranslator(frozenset({("cs", "en")}))
    cloud = FakeTranslator(frozenset({("cs", "de"), ("cs", "en")}), upper=True)
    pipe = GlyphFlipPipeline(FakeOcr(BOXES), local, (cloud,))

    result = pipe.process(b"fake-image", "en", source_language="cs")
    # primary supports it -> local used, cloud untouched
    assert len(local.calls) == 1
    assert len(cloud.calls) == 0
    assert result.translated_text == "<first>\n<second>\n<third>"


def test_process_falls_back_when_primary_cannot_serve() -> None:
    local = FakeTranslator(frozenset({("cs", "en")}))
    cloud = FakeTranslator(frozenset({("cs", "de")}), upper=True)
    pipe = GlyphFlipPipeline(FakeOcr(BOXES), local, (cloud,))

    result = pipe.process(b"fake-image", "de", source_language="cs")
    assert result.translated_text == "FIRST\nSECOND\nTHIRD"


def test_process_raises_when_nothing_supports_pair() -> None:
    local = FakeTranslator(frozenset({("cs", "en")}))
    pipe = GlyphFlipPipeline(FakeOcr(BOXES), local)

    with pytest.raises(UnsupportedPairError):
        pipe.process(b"fake-image", "ja", source_language="cs")


def test_segments_follow_reading_order() -> None:
    translator = FakeTranslator(frozenset({("en", "en")}))
    pipe = GlyphFlipPipeline(FakeOcr(BOXES), translator)
    pipe.process(b"x", "en", source_language="en")
    sent = translator.calls[0].segments
    assert sent == ("first", "second", "third")  # sorted by y, not dict order


def test_translation_failure_is_wrapped_with_stage() -> None:
    local = FakeTranslator(frozenset({("cs", "en")}), fail=True)
    pipe = GlyphFlipPipeline(FakeOcr(BOXES), local)

    with pytest.raises(PipelineError, match="translation stage failed"):
        pipe.process(b"x", "en", source_language="cs")


def test_no_text_found_raises_pipeline_error() -> None:
    pipe = GlyphFlipPipeline(FakeOcr(()), FakeTranslator(frozenset({("cs", "en")})))
    with pytest.raises(PipelineError, match="no text"):
        pipe.process(b"x", "en", source_language="cs")


def test_extract_text_wraps_ocr_errors() -> None:
    class ExplodingOcr(FakeOcr):
        def recognize(self, image, options):
            raise ValueError("bad image")

    pipe = GlyphFlipPipeline(ExplodingOcr(()), FakeTranslator(frozenset()))
    with pytest.raises(PipelineError, match="OCR stage failed"):
        pipe.extract_text(b"x")

