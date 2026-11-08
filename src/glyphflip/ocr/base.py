"""OCR backend abstraction.

Everything in glyphflip talks to an `OcrBackend`, never to a concrete
engine. This keeps the seam clean for swapping RapidOCR -> PaddleOCR ->
a custom engine later without touching the pipeline or CLI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

# Supported source-language tags. Backends map these to their own
# internal codes (e.g. "cs" -> "czech latin" in PP-OCR model names).
SUPPORTED_LANGUAGES = frozenset(
    {
        "en",  # english
        "cs",  # czech
        "de",  # german
        "fr",  # french
        "es",  # spanish
        "ru",  # russian (cyrillic)
        "uk",  # ukrainian (cyrillic)
        "zh",  # chinese simplified
        "ja",  # japanese
        "ko",  # korean
    }
)


@dataclass(frozen=True)
class TextBox:
    """One recognized region of text in the source image."""

    text: str
    #: axis-aligned bounding box as (x_min, y_min, x_max, y_max), pixels
    bbox: tuple[float, float, float, float]
    #: engine confidence in [0.0, 1.0]; None if the backend cannot estimate
    confidence: float | None = None

    def below(self, other: TextBox, tolerance: float = 5.0) -> bool:
        """True if this box sits vertically below `other` (same column-ish).

        Used by line-reconstruction: OCR engines emit boxes per detected
        region, and reading order is not guaranteed. Tolerance absorbs
        ascender/descender jitter between adjacent lines.
        """
        return self.bbox[1] > other.bbox[3] - tolerance


@dataclass(frozen=True)
class OcrResult:
    """Full OCR output for one image."""

    boxes: tuple[TextBox, ...] = ()

    @property
    def raw_text(self) -> str:
        """Naive reading order: sort top-to-bottom, left-to-right."""
        ordered = sorted(
            self.boxes,
            key=lambda b: (round(b.bbox[1], 1), b.bbox[0]),
        )
        return "\n".join(b.text for b in ordered)

    def mean_confidence(self) -> float | None:
        vals = [b.confidence for b in self.boxes if b.confidence is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)


@dataclass(frozen=True)
class OcrOptions:
    """Knobs common to every backend; engines ignore what they don't support."""

    language: str = "en"
    #: minimum [0.0, 1.0]; boxes below this are dropped
    min_confidence: float = 0.4
    #: per-backend passthrough (e.g. {"det_db_thresh": 0.3})
    engine_hints: dict[str, str | int | float] = field(default_factory=dict)


class OcrBackend(ABC):
    """Base class for OCR engines.

    Implementations must be lazy: heavy models are loaded on first
    `recognize()` call, not in `__init__`, so constructing a backend is
    cheap and side-effect free (keeps tests and `--help` fast).
    """

    #: registry key, e.g. "rapidocr" / "paddleocr" / "custom"
    name: str = "base"

    #: language tags this backend actually ships models for
    languages: frozenset[str] = frozenset()

    @abstractmethod
    def recognize(self, image: Path | bytes, options: OcrOptions) -> OcrResult:
        """OCR one image and return boxes in image coordinates.

        Args:
            image: path to an image file, or raw encoded bytes.
            options: see OcrOptions; language must be in SUPPORTED_LANGUAGES.

        Raises:
            UnsupportedLanguageError: options.language is not handled.
            ImageLoadError: the input is not decodable as an image.
        """

    @abstractmethod
    def warm_up(self, language: str) -> None:
        """Force-load models for `language` (called before batch runs)."""

    def supports(self, language: str) -> bool:
        return language in self.languages


class OcrError(Exception):
    """Base class for all OCR failures."""


class UnsupportedLanguageError(OcrError):
    def __init__(self, language: str, backend: str) -> None:
        super().__init__(
            f"backend {backend!r} has no model for language {language!r}"
        )
        self.language = language
        self.backend = backend


class ImageLoadError(OcrError):
    pass


def clamp_options(options: OcrOptions) -> OcrOptions:
    """Validate + normalize shared knobs before they reach a backend."""
    lang = options.language.lower()
    if lang not in SUPPORTED_LANGUAGES:
        raise UnsupportedLanguageError(lang, "options")
    return OcrOptions(
        language=lang,
        min_confidence=min(max(options.min_confidence, 0.0), 1.0),
        engine_hints=options.engine_hints,
    )

