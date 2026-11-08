"""Cloud translation backend (DeepL-style REST API, httpx).

The fallback leg of the hybrid: serves pairs the local opus-mt can't
(e.g. cs->de), configured via --api-key or GF_CLOUD_KEY env var.

Provider-neutral by design: DeepL and DeepL-compatible endpoints are the
v1 target (freetier api-free.deepl.com, pro api.deepl.com). Other vendors
(Google, Azure) get their own class implementing Translator later — same
seam as everywhere else in glyphflip.

License note: this is our own thin HTTP client; no SDK dependency, no
license entanglement. Network tests are marked `network` and skipped by
default.
"""

from __future__ import annotations

import logging

import httpx

from glyphflip.translate.base import (
    TranslationContentError,
    TranslationRequest,
    TranslationResult,
    Translator,
    TranslatorError,
    UnsupportedPairError,
    validate_request,
)

logger = logging.getLogger(__name__)

_DEEPL_HOSTS = {
    "free": "https://api-free.deepl.com",
    "pro": "https://api.deepl.com",
}
_TIMEOUT_S = 30.0

#: our tags -> DeepL language codes (uppercase ISO 639-1; regional
#: variants like en-US exist but plain codes are accepted as aliases)
_DEEPL_CODES: dict[str, str] = {
    "en": "EN",
    "cs": "CS",
    "de": "DE",
    "fr": "FR",
    "es": "ES",
    "it": "IT",
    "pl": "PL",
    "pt": "PT",
    "nl": "NL",
    "sv": "SV",
    "ru": "RU",
    "uk": "UK",
    "zh": "ZH",
    "ja": "JA",
    "ko": "KO",
    "tr": "TR",
    "ar": "AR",
    "hi": None,  # DeepL has no hindi in v1; None = unsupported
}


class CloudTranslator(Translator):
    """DeepL-compatible REST translator. Requires an API key."""

    name = "deepl"

    def __init__(self, api_key: str, *, tier: str = "free") -> None:
        if not api_key:
            raise TranslatorError("cloud translator requires an API key")
        self._api_key = api_key
        if tier not in _DEEPL_HOSTS:
            raise TranslatorError(f"unknown tier {tier!r}; use 'free' or 'pro'")
        self._base_url = _DEEPL_HOSTS[tier]

    # ------------------------------------------------------------------
    # capability
    # ------------------------------------------------------------------

    def supports(self, source: str | None, target: str) -> bool:
        # target must be servable; source can be anything DeepL can
        # autodetect, so None is fine here (unlike opus-mt v1)
        return _DEEPL_CODES.get(target) is not None

    def warm_up(self, pair: tuple[str, str]) -> None:
        # usage ping doubles as auth check; cheap, no model to load
        self._usage()

    # ------------------------------------------------------------------
    # translation
    # ------------------------------------------------------------------

    def translate(self, request: TranslationRequest) -> TranslationResult:
        req = validate_request(request)
        target_code = _DEEPL_CODES.get(req.target_language)
        if target_code is None:
            raise UnsupportedPairError(
                req.source_language, req.target_language, self.name
            )

        payload: dict = {
            "text": list(req.segments),  # DeepL accepts arrays: one result each
            "target_lang": target_code,
        }
        if req.source_language is not None:
            src = _DEEPL_CODES.get(req.source_language)
            if src is None:
                raise UnsupportedPairError(
                    req.source_language, req.target_language, self.name
                )
            payload["source_lang"] = src

        data = self._post("/v2/translate", json=payload)
        translations = data.get("translations", [])
        if len(translations) != len(req.segments):
            raise TranslationContentError(
                f"provider returned {len(translations)} translations "
                f"for {len(req.segments)} segments"
            )

        detected = req.source_language
        if detected is None and translations:
            detected = str(translations[0].get("detected_source_language", "")).lower()
            detected = detected or "und"

        return TranslationResult(
            translations=tuple(str(t["text"]) for t in translations),
            source_language=detected or "und",
            target_language=req.target_language,
        )

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"DeepL-Auth-Key {self._api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, json: dict) -> dict:
        try:
            resp = httpx.post(
                f"{self._base_url}{path}",
                headers=self._headers(),
                json=json,
                timeout=_TIMEOUT_S,
            )
        except httpx.HTTPError as e:
            raise TranslatorError(f"cloud provider unreachable: {e}") from e
        if resp.status_code == 403:
            raise TranslatorError("cloud provider rejected the API key (403)")
        if resp.status_code == 429:
            raise TranslatorError("cloud provider rate limit hit (429)")
        if resp.status_code >= 400:
            raise TranslatorError(
                f"cloud provider error {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()

    def _usage(self) -> dict:
        return self._post("/v2/usage", json={})

