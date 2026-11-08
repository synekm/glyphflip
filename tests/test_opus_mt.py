"""Tests for the Opus-MT translator.

Split in two tiers:
- fast (default): pure logic — pair resolution, batching, validation,
  unsupported pairs. No torch model is ever loaded here; we test the
  translator's decision layer by construction only.
- slow (-m slow): one real translation round-trip with the actual
  MarianMT model. First run downloads ~300MB to the HF cache.
"""

from __future__ import annotations

import pytest

from glyphflip.translate.base import (
    TranslationContentError,
    TranslationRequest,
    TranslationResult,
    UnsupportedPairError,
    validate_request,
)
from glyphflip.translate.opus_mt import (
    OpusMtTranslator,
    _batches,
    repo_for_pair,
)

# ---------------------------------------------------------------------------
# fast tier: pure logic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "target", "repo"),
    [
        ("cs", "en", "Helsinki-NLP/opus-mt-cs-en"),
        ("en", "de", "Helsinki-NLP/opus-mt-en-de"),
        ("ru", "en", "Helsinki-NLP/opus-mt-ru-en"),
        ("ja", "en", "Helsinki-NLP/opus-mt-ja-en"),
    ],
)
def test_repo_for_known_pairs(source: str, target: str, repo: str) -> None:
    assert repo_for_pair(source, target) == repo


def test_repo_for_unknown_pair_is_none() -> None:
    assert repo_for_pair("cs", "de") is None


def test_translator_supports_matrix() -> None:
    t = OpusMtTranslator()
    assert t.supports("cs", "en")
    assert t.supports("en", "cs")
    assert not t.supports("cs", "de")  # no direct model in v1
    assert not t.supports(None, "en")  # v1 needs explicit source


def test_validate_request_rejects_bad_target() -> None:
    with pytest.raises(UnsupportedPairError):
        validate_request(TranslationRequest(("ahoj",), "cs", "xx"))


def test_validate_request_rejects_same_source_and_target() -> None:
    with pytest.raises(UnsupportedPairError):
        validate_request(TranslationRequest(("ahoj",), "cs", "cs"))


def test_validate_request_rejects_empty_segments() -> None:
    with pytest.raises(TranslationContentError):
        validate_request(TranslationRequest((), "cs", "en"))


def test_validate_request_normalizes_case() -> None:
    req = validate_request(TranslationRequest(("ahoj",), "CS", "EN"))
    assert req.source_language == "cs"
    assert req.target_language == "en"


def test_batches_keeps_order_and_covers_all() -> None:
    segs = tuple(f"segment number {i}" for i in range(57))
    batches = _batches(segs)
    flat = [s for b in batches for s in b]
    assert flat == list(segs)  # order preserved, nothing lost


def test_batches_splits_large_inputs() -> None:
    segs = ("word " * 400,) * 20  # ~2000 chars each
    batches = _batches(segs)
    assert len(batches) > 1  # would blow the token budget in one call
    for b in batches:
        assert len(b) >= 1


def test_batches_handles_short_input() -> None:
    assert _batches(()) == []
    assert len(_batches(("hi",))) == 1


def test_result_merged_joins_lines() -> None:
    r = TranslationResult(("Hello", "world"), "cs", "en")
    assert r.merged() == "Hello\nworld"


# ---------------------------------------------------------------------------
# slow tier: real model
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cs_to_en_roundtrip() -> None:
    """Translate a known czech sentence; opus-mt cs-en is a strong pair."""
    t = OpusMtTranslator()
    req = validate_request(
        TranslationRequest(
            ("Dobrý den, jak se máte?",),
            "cs",
            "en",
        )
    )
    result = t.translate(req)
    joined = result.translations[0].lower()
    assert "hello" in joined or "good day" in joined
    assert "how" in joined


@pytest.mark.slow
def test_multi_segment_alignment() -> None:
    """Each input segment yields exactly one output, in order."""
    t = OpusMtTranslator()
    segs = ("Dobrý den.", "Jak se máte?", "Děkuji.")
    req = validate_request(
        TranslationRequest(segs, "cs", "en")
    )
    result = t.translate(req)
    assert len(result.translations) == len(segs)
    low = [x.lower() for x in result.translations]
    assert any("thank" in x for x in low)  # "Děkuji." should mention thanks


@pytest.mark.slow
def test_warm_up_then_translate() -> None:
    t = OpusMtTranslator()
    t.warm_up(("en", "cs"))
    req = validate_request(TranslationRequest(("Good morning.",), "en", "cs"))
    result = t.translate(req)
    assert len(result.translations) == 1
    assert result.translations[0]  # non-empty

