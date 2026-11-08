"""Translation backend abstraction.

Same seam philosophy as glyphflip.ocr.base: the pipeline and CLI talk to a
`Translator`, never to opus-mt / cloud APIs / a custom engine directly.

Design notes:
- Translators receive *segments* (one per OCR box) rather than one blob, so
  they can batch efficiently and preserve line structure.
- Language tags reuse glyphflip's two-letter codes and are mapped to
  engine-specific codes by each backend (e.g. "cs" -> "ces" in NLLB/opus).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

#: language tags accepted as translation targets across the whole app
TRANSLATION_LANGUAGES = frozenset(
    {
        "en",  # english
        "cs",  # czech
        "de",  # german
        "fr",  # french
        "es",  # spanish
        "it",  # italian
        "pl",  # polish
        "pt",  # portuguese
        "nl",  # dutch
        "sv",  # swedish
        "ru",  # russian
        "uk",  # ukrainian
        "zh",  # chinese
        "ja",  # japanese
        "ko",  # korean
        "tr",  # turkish
        "ar",  # arabic
        "hi",  # hindi
    }
)


class TranslatorError(Exception):
    """Base class for translation failures."""


class UnsupportedPairError(TranslatorError):
    def __init__(self, source: str, target: str, backend: str) -> None:
        super().__init__(
            f"backend {backend!r} cannot translate {source!r} -> {target!r}"
        )
        self.source = source
        self.target = target
        self.backend = backend


class TranslationContentError(TranslatorError):
    """Engine ran but returned unusable output (empty/missing segments)."""


@dataclass(frozen=True)
class TranslationRequest:
    """One batch of segments to translate together.

    Segments, not a blob: OCR boxes stay independently addressable so the
    pipeline can reassemble translated text in the original layout.
    """

    segments: tuple[str, ...]
    source_language: str | None = None  # None = detect
    target_language: str = "en"


@dataclass(frozen=True)
class TranslationResult:
    """Per-segment translations, aligned with the request."""

    translations: tuple[str, ...]
    source_language: str  # resolved (detected or confirmed)
    target_language: str

    def merged(self) -> str:
        """Newline-joined, for CLI output when layout doesn't matter."""
        return "\n".join(self.translations)


class Translator(ABC):
    """Base class for translation engines.

    Same laziness rule as OcrBackend: __init__ is cheap; models/clients
    load on first translate() so imports stay side-effect free.
    """

    #: registry key: "opus-mt" / "cloud" / ...
    name: str = "base"

    #: (source, target) pairs this backend can serve. Empty means "ask me".
    pairs: frozenset[tuple[str, str]] = frozenset()

    @abstractmethod
    def translate(self, request: TranslationRequest) -> TranslationResult:
        """Translate every segment; len(translations) == len(segments).

        Raises:
            UnsupportedPairError: this backend can't serve the pair.
            TranslatorError: engine-level failure (network, OOM, etc.).
        """

    @abstractmethod
    def warm_up(self, pair: tuple[str, str]) -> None:
        """Force-load models/clients for a pair (before batch runs)."""

    def supports(self, source: str | None, target: str) -> bool:
        """Whether this backend claims to serve source->target.

        source=None (autodetect) is conservatively True: we only learn the
        real answer by trying. Backends with explicit pairs override this.
        """
        if source is None:
            return target in TRANSLATION_LANGUAGES
        return (source, target) in self.pairs


def validate_request(request: TranslationRequest) -> TranslationRequest:
    """Normalize + validate before a backend sees the request."""
    target = request.target_language.lower()
    if target not in TRANSLATION_LANGUAGES:
        raise UnsupportedPairError(
            request.source_language or "?", target, "request"
        )
    if not request.segments:
        raise TranslationContentError("no segments to translate")
    source = request.source_language
    if source is not None:
        source = source.lower()
        if source not in TRANSLATION_LANGUAGES:
            raise UnsupportedPairError(source, target, "request")
        if source == target:
            raise UnsupportedPairError(source, target, "request")
    return TranslationRequest(
        segments=request.segments,
        source_language=source,
        target_language=target,
    )

