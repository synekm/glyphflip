"""Opus-MT model resolution: drop-in dir > executable-adjacent > HF cache.

Drop-in layout (a directory, wherever the user unzipped a pair bundle):

    models/opus-mt-cs-en/{config.json, pytorch_model.bin, ...}

Priority:
    1. $GF_MODEL_DIR/models/<pair-dir>          (explicit override)
    2. <dir-of-executable>/models/<pair-dir>    (drop-in next to binary)
    3. HF hub repo "Helsinki-NLP/opus-mt-<pair>" (normal from_pretrained)

This module never imports torch — resolution is pure path logic, so the
cloud-only edition ships it without dragging heavyweight deps.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_PAIR_DIR_PREFIX = "opus-mt-"


def _candidate_roots() -> list[Path]:
    """Roots to scan, in priority order. Each holds models/<pair-dir>."""
    roots: list[Path] = []

    env = os.environ.get("GF_MODEL_DIR")
    if env:
        roots.append(Path(env))

    # directory containing the running executable (frozen or venv) —
    # drop-ins unzipped next to the binary are found automatically.
    if getattr(sys, "frozen", False):  # pyinstaller
        roots.append(Path(sys.executable).parent)
    else:
        # dev/pip mode: repo-root heuristic — walk up from this file
        # to the checkout root holding pyproject.toml
        pkg_dir = Path(__file__).resolve()
        for ancestor in pkg_dir.parents:
            if (ancestor / "pyproject.toml").is_file():
                roots.append(ancestor)
                break

    return roots


def resolve_model_path(source: str, target: str) -> str | None:
    """Return a local path to the model for source->target, or None.

    A usable drop-in must contain config.json (the minimum viable model
    dir for from_pretrained); missing files (weights, vocab) will then
    produce a clear transformers error rather than silent HF fallback.
    """
    pair_dir = f"{_PAIR_DIR_PREFIX}{source}-{target}"

    for root in _candidate_roots():
        candidate = root / "models" / pair_dir
        if (candidate / "config.json").is_file():
            logger.info("using drop-in model: %s", candidate)
            return str(candidate)
    return None


def list_installed_pairs() -> list[dict[str, str]]:
    """Inventory of drop-in models found, for `gf models` and debugging."""
    found: list[dict[str, str]] = []
    seen: set[Path] = set()
    for root in _candidate_roots():
        models = root / "models"
        if not models.is_dir() or models in seen:
            continue
        seen.add(models)
        for d in sorted(models.iterdir()):
            if d.is_dir() and d.name.startswith(_PAIR_DIR_PREFIX):
                pair = d.name[len(_PAIR_DIR_PREFIX):]
                if "-" in pair:
                    src, tgt = pair.split("-", 1)
                    found.append({"pair": pair, "source": src, "target": tgt,
                                  "path": str(d)})
    return found


def install_from_dir(pair: str, src: Path, dest_root: Path) -> Path:
    """Copy an unpacked model dir into a drop-in location (used by the
    bundle builder for validation, and handy for manual installs)."""
    dest = dest_root / "models" / f"{_PAIR_DIR_PREFIX}{pair}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return dest

