"""Contract tests for the edition split (cloud/local/all).

These lock the exact bug found in the first cloud build: the CLI must
consult opus_mt._TORCH_AVAILABLE (not import success) when deciding
whether a local translator leg exists. They run entirely in the fast
tier — no torch model is ever loaded, only the availability flag is
flipped.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# _build_translators contract
# ---------------------------------------------------------------------------


@pytest.fixture()
def no_torch(monkeypatch):
    """Simulate the cloud edition: torch/transformers absent."""
    from glyphflip.translate import opus_mt

    monkeypatch.setattr(opus_mt, "_TORCH_AVAILABLE", False)


@pytest.fixture()
def with_torch(monkeypatch):
    """Simulate the local/all edition."""
    from glyphflip.translate import opus_mt

    monkeypatch.setattr(opus_mt, "_TORCH_AVAILABLE", True)


def test_cloud_edition_no_local_leg(no_torch):
    from glyphflip.cli import _build_translators

    translators = _build_translators(api_key=None)
    # no torch + no key -> nothing available, but NO crash and no
    # phantom opus-mt leg that would claim support and then fail
    assert translators == ()


def test_cloud_edition_with_key(no_torch):
    from glyphflip.cli import _build_translators
    from glyphflip.translate.cloud import CloudTranslator

    translators = _build_translators(api_key="test-key")
    assert len(translators) == 1
    assert isinstance(translators[0], CloudTranslator)


def test_local_edition_has_opus_first(with_torch):
    from glyphflip.cli import _build_translators
    from glyphflip.translate.opus_mt import OpusMtTranslator

    translators = _build_translators(api_key=None)
    assert len(translators) == 1
    assert isinstance(translators[0], OpusMtTranslator)


def test_all_edition_order(with_torch):
    """With both legs, local is primary and cloud is fallback — in order."""
    from glyphflip.cli import _build_translators
    from glyphflip.translate.cloud import CloudTranslator
    from glyphflip.translate.opus_mt import OpusMtTranslator

    translators = _build_translators(api_key="test-key")
    assert [type(t) for t in translators] == [OpusMtTranslator, CloudTranslator]


# ---------------------------------------------------------------------------
# degraded messaging reaches the user
# ---------------------------------------------------------------------------


def test_cloud_edition_warns_about_missing_local_leg(no_torch, capsys):
    from glyphflip.cli import _build_translators

    _build_translators(api_key=None)
    err = capsys.readouterr().err
    assert "local translator not available" in err
    assert "no translator available" in err


def test_torch_flag_reflects_reality():
    """Sanity: in this dev env torch IS installed — the inverse contract
    to no_torch, guarding against the flag flipping meaning entirely."""
    from glyphflip.translate import opus_mt

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed in this environment")
    assert opus_mt._TORCH_AVAILABLE is True

