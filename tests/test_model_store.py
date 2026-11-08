"""Tests for the drop-in model store/resolver.

Pure path logic — no torch, no models downloaded. Fake model dirs are
built from empty files; the resolver only checks config.json existence,
so that's enough to test discovery in full.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from glyphflip.translate.model_store import (
    _candidate_roots,
    install_from_dir,
    list_installed_pairs,
    resolve_model_path,
)


def _make_fake_model(root: Path, pair: str, *, complete: bool = True) -> Path:
    """Create models/opus-mt-<pair>/{config.json[, vocab.json, weights]}."""
    d = root / "models" / f"opus-mt-{pair}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}")
    if complete:
        (d / "vocab.json").write_text("{}")
        (d / "pytorch_model.bin").write_bytes(b"\x00" * 16)
    return d


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def test_env_var_root_wins(monkeypatch, tmp_path):
    store_a, store_b = tmp_path / "a", tmp_path / "b"
    _make_fake_model(store_a, "cs-en")
    _make_fake_model(store_b, "cs-en")

    # GF_MODEL_DIR points at store_a; find_spec's package-parent heuristic
    # could theoretically find store_b — env var must take priority.
    monkeypatch.setenv("GF_MODEL_DIR", str(store_a))
    got = resolve_model_path("cs", "en")
    assert got is not None
    assert got.startswith(str(store_a.resolve()))


def test_env_var_root_found_without_config_json_skipped(monkeypatch, tmp_path):
    store = tmp_path / "store"
    d = store / "models" / "opus-mt-cs-en"
    d.mkdir(parents=True)
    (d / "vocab.json").write_text("{}")  # no config.json -> unusable

    monkeypatch.setenv("GF_MODEL_DIR", str(store))
    assert resolve_model_path("cs", "en") is None


def test_missing_pair_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("GF_MODEL_DIR", str(tmp_path))
    assert resolve_model_path("cs", "en") is None


def test_repo_root_heuristic_dev_mode(tmp_path):
    """In non-frozen mode, a pyproject.toml-bearing ancestor is a root."""
    # findspec-based heuristic is exercised implicitly everywhere in dev;
    # here we assert a root containing pyproject.toml is actually scanned.
    roots = _candidate_roots()
    assert any((r / "pyproject.toml").is_file() for r in roots), (
        "dev-mode candidate roots must include the project root"
    )


def test_frozen_mode_uses_executable_dir(monkeypatch, tmp_path):
    fake_exe_dir = tmp_path / "bin"
    fake_exe_dir.mkdir()
    (fake_exe_dir / "gf").write_text("#!/bin/sh\n")
    _make_fake_model(fake_exe_dir, "cs-en")

    import sys

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe_dir / "gf"))
    monkeypatch.delenv("GF_MODEL_DIR", raising=False)

    got = resolve_model_path("cs", "en")
    assert got is not None
    assert str(fake_exe_dir) in got


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


def test_list_installed_pairs(monkeypatch, tmp_path):
    store = tmp_path / "store"
    _make_fake_model(store, "cs-en")
    _make_fake_model(store, "en-de")
    # noise that must be ignored:
    (store / "models" / "random-dir").mkdir()
    (store / "models" / "opus-mt-broken").mkdir()  # no '-' separator... actually invalid name shape

    monkeypatch.setenv("GF_MODEL_DIR", str(store))
    pairs = list_installed_pairs()
    found = {p["pair"] for p in pairs}
    assert found == {"cs-en", "en-de"}


def test_list_deduplicates_overlapping_roots(monkeypatch, tmp_path):
    """Same model visible via two roots -> reported once."""
    store = tmp_path / "store"
    _make_fake_model(store, "cs-en")

    monkeypatch.setenv("GF_MODEL_DIR", str(store))
    import sys

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    fake_exe_dir = store  # same directory through the frozen path too
    monkeypatch.setattr(sys, "executable", str(fake_exe_dir / "gf"))

    pairs = list_installed_pairs()
    assert sum(1 for p in pairs if p["pair"] == "cs-en") == 1


# ---------------------------------------------------------------------------
# installer helper
# ---------------------------------------------------------------------------


def test_install_from_dir_copies_and_cleans(monkeypatch, tmp_path):
    src = tmp_path / "src" / "opus-mt-cs-en"
    src.mkdir(parents=True)
    (src / "config.json").write_text('{"old": true}')

    dest_root = tmp_path / "dest"
    dest_root.mkdir()

    first = install_from_dir("cs-en", src, dest_root)
    assert (first / "config.json").read_text() == '{"old": true}'

    # second install overwrites cleanly
    (src / "config.json").write_text('{"new": true}')
    second = install_from_dir("cs-en", src, dest_root)
    assert second == first
    assert (second / "config.json").read_text() == '{"new": true}'

