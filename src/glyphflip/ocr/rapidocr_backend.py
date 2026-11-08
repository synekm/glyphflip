"""RapidOCR adapter (v3.x `rapidocr`, PP-OCRv6 defaults; 1.x fallback kept).

Single seam responsibility: normalize whatever RapidOCR returns into our
TextBox/OcrResult types.

Handles two generations of output shapes:
- rapidocr 3.x: `RapidOCROutput` with .boxes (N,4,2 polygons), .txts, .scores
- rapidocr_onnxruntime 1.x: list of [points, text, score] (or None)
"""

from __future__ import annotations

import logging
from pathlib import Path

from glyphflip.imaging.loader import load_image
from glyphflip.imaging.preprocess import maybe_upscale
from glyphflip.ocr.base import (
    ImageLoadError,
    OcrBackend,
    OcrOptions,
    OcrResult,
    TextBox,
    UnsupportedLanguageError,
    clamp_options,
)

logger = logging.getLogger(__name__)

try:  # rapidocr >= 2.x
    from rapidocr import RapidOCR as _RapidOCR
except ImportError:  # pragma: no cover - legacy 1.x package
    try:
        from rapidocr_onnxruntime import RapidOCR as _RapidOCR
    except ImportError as e:
        raise ImportError(
            "rapidocr is not installed; it is a core dependency of glyphflip. "
            "Run: uv pip install -e '.'"
        ) from e

__all__ = ["RapidocrBackend", "ImageLoadError", "UnsupportedLanguageError"]


def _polygon_to_bbox(points: object) -> tuple[float, float, float, float]:
    """Convert a 4-point polygon [[x,y],...] to an AABB (xmin, ymin, xmax, ymax)."""
    xs = [float(p[0]) for p in points]  # type: ignore[index]
    ys = [float(p[1]) for p in points]  # type: ignore[index]
    return min(xs), min(ys), max(xs), max(ys)


class RapidocrBackend(OcrBackend):
    """Default backend. Bundled PP-OCRv6 det/rec models (rapidocr >= 3.10).

    Small inputs (screenshots of UI text) are upscaled before recognition
    — tiny fonts are the #1 accuracy killer in our use case.
    """

    name = "rapidocr"
    languages = frozenset({"en", "cs", "de", "fr", "es", "ru", "uk"})
    # NOTE: PP-OCRv6's default multilingual rec handles latin+cyrillic
    # scripts incl. full diacritics (czech verified empirically).
    # Language-specific rec models (LangRec.ESLAV etc.) can be injected
    # via engine_hints when we want per-language tuning.

    def __init__(self, *, upscale_min_side: int | None = 800) -> None:
        self._engine: _RapidOCR | None = None
        self._upscale_min_side = upscale_min_side  # None disables

    def _get_engine(self) -> _RapidOCR:
        if self._engine is None:
            logger.info("loading rapidocr engine (first use)...")
            self._engine = _RapidOCR()
            # rapidocr forces its "RapidOCR" logger to INFO with its own
            # colorlog handler at import/init time — AFTER our CLI config
            # ran (engine import is deliberately lazy). Re-applying the
            # mute here wins the race: level is checked before any
            # handler fires, so this silences it regardless of the
            # handler it installed.
            logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        return self._engine

    def warm_up(self, language: str) -> None:
        self._get_engine()

    def recognize(self, image: Path | bytes, options: OcrOptions) -> OcrResult:
        opts = clamp_options(options)
        if not self.supports(opts.language):
            raise UnsupportedLanguageError(opts.language, self.name)

        arr = load_image(image)  # np.ndarray, RGB
        if self._upscale_min_side is not None:
            arr = maybe_upscale(arr, self._upscale_min_side)
        engine = self._get_engine()

        raw = engine(arr)
        # 1.x returned (result, elapsed); 3.x returns a single object
        if isinstance(raw, tuple) and len(raw) == 2:
            raw = raw[0]

        boxes: list[TextBox] = []

        # --- rapidocr >= 3.x: attribute-style output object ---------------
        if hasattr(raw, "txts") and raw.txts is not None and len(raw.txts):
            for pts, text, score in zip(raw.boxes, raw.txts, raw.scores):
                score = float(score)
                if score < opts.min_confidence:
                    continue
                boxes.append(
                    TextBox(
                        text=str(text).strip(),
                        bbox=_polygon_to_bbox(pts),
                        confidence=score,
                    )
                )
            return OcrResult(boxes=tuple(boxes))

        # --- legacy 1.x: list of [points, text, score] or None ------------
        if raw is None:  # no text found
            return OcrResult(boxes=())
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                pts, text, score = item[0], str(item[1]), float(item[2])
            elif isinstance(item, dict):
                pts, text, score = item["box"], str(item["text"]), float(item["score"])
            else:  # pragma: no cover - defensive
                continue
            if score < opts.min_confidence:
                continue
            boxes.append(TextBox(text=text, bbox=_polygon_to_bbox(pts), confidence=score))

        return OcrResult(boxes=tuple(boxes))

