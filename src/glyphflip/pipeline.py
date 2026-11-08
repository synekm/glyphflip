"""Pipeline: image -> OCR -> translation.

Owns the hybrid routing rule (which translator serves a given pair) and
the orchestration that keeps translated segments aligned with the OCR
layout so callers can reconstruct page order.
"""

from __future__ import annotations

import logging
from pathlib import Path

from glyphflip.ocr.base import (
    OcrBackend,
    OcrOptions,
    OcrResult,
    TextBox,
)
from glyphflip.translate.base import (
    TranslationRequest,
    TranslationResult,
    Translator,
    UnsupportedPairError,
)

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """End-to-end failure: wraps engine errors with stage context."""


class GlyphFlipPipeline:
    """image -> OcrBackend -> Translator -> translated segments in layout order.

    Hybrid routing in v1 is simple and explicit:
      - one primary translator (default: opus-mt, local)
      - zero or more fallbacks (default: none; the cloud translator slots
        in here in file #16)
    The first backend whose `supports()` returns True wins. If none do,
    we surface the primary's UnsupportedPairError to the caller.
    """

    def __init__(
        self,
        ocr_backend: OcrBackend,
        primary_translator: Translator,
        fallback_translators: tuple[Translator, ...] = (),
    ) -> None:
        self._ocr = ocr_backend
        self._primary = primary_translator
        self._fallbacks = tuple(fallback_translators)

    # ------------------------------------------------------------------
    # routing
    # ------------------------------------------------------------------

    def _route(self, source: str | None, target: str) -> tuple[Translator, ...]:
        """Translators to try, in order, for this pair."""
        return (self._primary, *self._fallbacks)

    def _pick_translator(self, source: str | None, target: str) -> Translator:
        for translator in self._route(source, target):
            if translator.supports(source, target):
                return translator
        names = ", ".join(t.name for t in self._route(source, target))
        raise UnsupportedPairError(source or "auto", target, names)

    # ------------------------------------------------------------------
    # stages
    # ------------------------------------------------------------------

    def extract_text(
        self, image: Path | bytes, ocr_options: OcrOptions | None = None
    ) -> OcrResult:
        """Stage 1: raw OCR. Errors wrapped with stage context."""
        options = ocr_options or OcrOptions()
        try:
            return self._ocr.recognize(image, options)
        except Exception as e:
            raise PipelineError(f"OCR stage failed: {e}") from e

    def translate_text(
        self,
        ocr_result: OcrResult,
        target_language: str,
        source_language: str | None = None,
    ) -> TranslationResult:
        """Stage 2: translate OCR output, one segment per box.

        Boxes are translated in reading order (same sort as raw_text) so
        index i of the result corresponds to the i-th laid-out line.
        """
        segments = tuple(
            b.text
            for b in _reading_order(ocr_result.boxes)
            if b.text.strip()
        )
        if not segments:
            raise PipelineError("OCR found no text to translate")

        request = TranslationRequest(
            segments=segments,
            source_language=source_language,
            target_language=target_language,
        )

        translator = self._pick_translator(source_language, target_language)
        try:
            return translator.translate(request)
        except UnsupportedPairError:
            raise  # routing lied; let the caller see it plainly
        except Exception as e:
            raise PipelineError(
                f"translation stage failed ({translator.name}): {e}"
            ) from e

    # ------------------------------------------------------------------
    # end-to-end
    # ------------------------------------------------------------------

    def process(
        self,
        image: Path | bytes,
        target_language: str,
        *,
        source_language: str | None = None,
        ocr_options: OcrOptions | None = None,
    ) -> "GlyphFlipResult":
        """Full run: OCR then translate. The `gf translate` codepath."""
        ocr_result = self.extract_text(image, ocr_options)
        translation = self.translate_text(
            ocr_result, target_language, source_language
        )
        return GlyphFlipResult(
            ocr=ocr_result,
            translation=translation,
            ordered_boxes=_reading_order(ocr_result.boxes),
        )


class GlyphFlipResult:
    """Everything one `process()` call produced, layout-aware."""

    def __init__(
        self, ocr: OcrResult, translation: TranslationResult,
        ordered_boxes: tuple[TextBox, ...],
    ) -> None:
        self.ocr = ocr
        self.translation = translation
        # boxes in reading order, same ordering used to build segments,
        # so translation.translations[i] describes ordered_boxes[i] --
        # minus any blank boxes filtered out during segmentation.
        self.ordered_boxes = ordered_boxes

    @property
    def translated_text(self) -> str:
        """Newline-joined translations, reading order."""
        return self.translation.merged()

    @property
    def source_text(self) -> str:
        return self.ocr.raw_text


def _reading_order(boxes: tuple[TextBox, ...]) -> tuple[TextBox, ...]:
    """Top-to-bottom, left-to-right — same as OcrResult.raw_text."""
    return tuple(sorted(boxes, key=lambda b: (round(b.bbox[1], 1), b.bbox[0])))

