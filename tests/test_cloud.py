"""Cloud translator tests — HTTP faked, no real calls.

Real-API tests would be `network`-marked; none yet (needs a key + CI
secrets story). Everything here monkeypatches httpx.post.
"""

from __future__ import annotations

import pytest

import glyphflip.translate.cloud as cloud_mod
from glyphflip.translate.base import (
    TranslationRequest,
    TranslatorError,
    UnsupportedPairError,
    validate_request,
)
from glyphflip.translate.cloud import CloudTranslator

KEY = "fake-key-123:fx"


class _FakeResponse:
    def __init__(self, status_code: int, data: dict | None = None,
                 text: str = "") -> None:
        self.status_code = status_code
        self._data = data or {}
        self.text = text

    def json(self) -> dict:
        return self._data


@pytest.fixture
def translator() -> CloudTranslator:
    return CloudTranslator(KEY)


def _ok_response(payload: dict) -> _FakeResponse:
    return _FakeResponse(200, payload)


def test_supports_target_matrix(translator: CloudTranslator) -> None:
    assert translator.supports("cs", "de")  # the pair opus-mt can't do!
    assert translator.supports(None, "de")  # autodetect works on cloud
    assert not translator.supports("cs", "hi")  # no hindi on DeepL
    assert not translator.supports("cs", "xx")


def test_requires_key() -> None:
    with pytest.raises(TranslatorError):
        CloudTranslator("")


def test_translate_sends_batch_and_parses(monkeypatch, translator) -> None:
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, json=json)
        return _ok_response({
            "translations": [
                {"text": "Entwürfe", "detected_source_language": "CS"},
                {"text": "Papierkorb", "detected_source_language": "CS"},
            ]
        })

    monkeypatch.setattr(cloud_mod.httpx, "post", fake_post)

    req = validate_request(TranslationRequest(("Rozepsané", "Koš"), "cs", "de"))
    result = translator.translate(req)

    assert result.translations == ("Entwürfe", "Papierkorb")
    assert result.source_language == "cs"
    assert result.target_language == "de"

    # request shape: array text, uppercase codes, auth header
    assert captured["json"]["text"] == ["Rozepsané", "Koš"]
    assert captured["json"]["target_lang"] == "DE"
    assert captured["json"]["source_lang"] == "CS"
    assert captured["headers"]["Authorization"].startswith("DeepL-Auth-Key")
    assert "v2/translate" in captured["url"]


def test_autodetect_reports_detected_language(monkeypatch, translator) -> None:
    def fake_post(url, headers=None, json=None, timeout=None):
        assert "source_lang" not in json  # must not pin source
        return _ok_response({
            "translations": [
                {"text": "Drafts", "detected_source_language": "CS"}
            ]
        })

    monkeypatch.setattr(cloud_mod.httpx, "post", fake_post)

    req = validate_request(TranslationRequest(("Rozepsané",), None, "en"))
    result = translator.translate(req)
    assert result.source_language == "cs"
    assert result.translations == ("Drafts",)


def test_unsupported_source_raises(monkeypatch, translator) -> None:
    req = validate_request(TranslationRequest(("नमस्ते",), "hi", "en"))
    with pytest.raises(UnsupportedPairError):
        translator.translate(req)


def test_misaligned_provider_response_raises(monkeypatch, translator) -> None:
    monkeypatch.setattr(
        cloud_mod.httpx, "post",
        lambda *a, **k: _ok_response({"translations": [{"text": "one"}]}),
    )
    req = validate_request(TranslationRequest(("a", "b"), "cs", "de"))
    with pytest.raises(cloud_mod.TranslationContentError):
        translator.translate(req)


def test_http_error_maps_to_translator_error(monkeypatch, translator) -> None:
    import httpx as _httpx

    def boom(*a, **k):
        raise _httpx.ConnectError("no route")

    monkeypatch.setattr(cloud_mod.httpx, "post", boom)
    req = validate_request(TranslationRequest(("ahoj",), "cs", "de"))
    with pytest.raises(TranslatorError, match="unreachable"):
        translator.translate(req)


@pytest.mark.parametrize(
    ("code", "match"),
    [(403, "rejected the API key"), (429, "rate limit"), (500, "provider error")],
)
def test_status_codes_mapped(monkeypatch, translator, code, match) -> None:
    monkeypatch.setattr(
        cloud_mod.httpx, "post",
        lambda *a, **k: _FakeResponse(code, text="boom"),
    )
    req = validate_request(TranslationRequest(("ahoj",), "cs", "de"))
    with pytest.raises(TranslatorError, match=match):
        translator.translate(req)

