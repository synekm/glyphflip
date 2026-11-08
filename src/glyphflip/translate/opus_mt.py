"""Opus-MT translation backend (Helsinki-NLP, CC-BY 4.0).

Local translation via a small per-language-pair MarianMT model. Models
resolve from a drop-in directory first (see model_store.py), else load
from the HF hub cache.

Torch-guarded: on the cloud-only edition (no torch installed), this
module imports fine but `supports()` returns False, so routing falls
through to the cloud translator.

Supported pairs in v1 are direct X->English and English->X models.
Non-English targets from non-English sources are served by the cloud leg.

License note: opus-mt models are CC-BY 4.0 — attribution required if WE
redistribute weights (NOTICE.md obligation #2); runtime HF fetch or the
user's own drop-in bundle carries no such duty for us.
"""

from __future__ import annotations

import logging

from glyphflip.translate.base import (
    TranslationContentError,
    TranslationRequest,
    TranslationResult,
    Translator,
    UnsupportedPairError,
    validate_request,
)

logger = logging.getLogger(__name__)

try:
    from transformers import MarianMTModel, MarianTokenizer
except ImportError:  # cloud edition: leg is inert, not broken
    MarianMTModel = None  # type: ignore[assignment,misc]
    MarianTokenizer = None  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False
else:
    _TORCH_AVAILABLE = True

#: rough cap on tokens per generate() batch
_MAX_BATCH_TOKENS = 512
#: crude chars-per-token heuristic for OCR-sized latin text
_CHARS_PER_TOKEN = 4


#: language pairs with known-good direct opus-mt models (bidirectional,
#: all via English as either source or target)
_DIRECT_PAIRS: frozenset[tuple[str, str]] = frozenset(
    [
        ("en", "cs"), ("cs", "en"),
        ("en", "de"), ("de", "en"),
        ("en", "fr"), ("fr", "en"),
        ("en", "es"), ("es", "en"),
        ("en", "ru"), ("ru", "en"),
        ("en", "uk"), ("uk", "en"),
        ("en", "zh"), ("zh", "en"),
        ("en", "ja"), ("ja", "en"),
        ("en", "ko"), ("ko", "en"),
        ("en", "pl"), ("pl", "en"),
        ("en", "tr"), ("tr", "en"),
        ("en", "hi"), ("hi", "en"),
        ("en", "it"), ("it", "en"),
        ("en", "nl"), ("nl", "en"),
        ("en", "pt"), ("pt", "en"),
        ("en", "sv"), ("sv", "en"),
        ("en", "fi"), ("fi", "en"),
    ]
)


def repo_for_pair(source: str, target: str) -> str | None:
    """Map a language pair to its HuggingFace repo, or None if unknown."""
    if (source, target) in _DIRECT_PAIRS:
        return f"Helsinki-NLP/opus-mt-{source}-{target}"
    return None


class OpusMtTranslator(Translator):
    """Local MarianMT (opus-mt) translation, one model per pair, cached."""

    name = "opus-mt"

    def __init__(self, *, device: str = "cpu") -> None:
        self._device = device
        self._models: dict[str, tuple[MarianTokenizer, MarianMTModel]] = {}

    def supports(self, source: str | None, target: str) -> bool:
        if not _TORCH_AVAILABLE:
            return False  # cloud edition: this leg is inert by design
        if source is None:
            return False  # v1 requires explicit source; autodetect later
        return repo_for_pair(source, target) is not None

    def warm_up(self, pair: tuple[str, str]) -> None:
        self._load(pair[0], pair[1])

    def _load(self, source: str, target: str) -> tuple[MarianTokenizer, MarianMTModel]:
        if not _TORCH_AVAILABLE:
            raise UnsupportedPairError(
                source, target, f"{self.name} (torch not installed — "
                "use the cloud translator or a local edition of glyphflip)"
            )
        repo = repo_for_pair(source, target)
        if repo is None:
            raise UnsupportedPairError(source, target, self.name)
        if repo not in self._models:
            from glyphflip.translate.model_store import resolve_model_path

            local = resolve_model_path(source, target)
            logger.info("loading %s (first use)...", local or repo)
            tokenizer = MarianTokenizer.from_pretrained(local or repo)
            model = MarianMTModel.from_pretrained(local or repo)
            model.eval()
            self._models[repo] = (tokenizer, model)
        return self._models[repo]

    def translate(self, request: TranslationRequest) -> TranslationResult:
        req = validate_request(request)
        if req.source_language is None:
            raise UnsupportedPairError(None, req.target_language, self.name)

        tokenizer, model = self._load(req.source_language, req.target_language)

        translations: list[str] = []
        for batch in _batches(req.segments):
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=_MAX_BATCH_TOKENS,
            )
            generated = model.generate(**encoded)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            translations.extend(decoded)

        if len(translations) != len(req.segments):
            raise TranslationContentError(
                f"got {len(translations)} translations "
                f"for {len(req.segments)} segments"
            )
        return TranslationResult(
            translations=tuple(t.strip() for t in translations),
            source_language=req.source_language,
            target_language=req.target_language,
        )


def _batches(segments: tuple[str, ...]) -> list[list[str]]:
    """Group segments so each batch stays under the token budget."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for seg in segments:
        est = max(1, len(seg) // _CHARS_PER_TOKEN) + 2  # + specials
        if current and current_tokens + est > _MAX_BATCH_TOKENS:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(seg)
        current_tokens += est
    if current:
        batches.append(current)
    return batches

