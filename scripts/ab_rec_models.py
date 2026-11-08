#!/usr/bin/env python3
"""A/B harness: grade OCR rec models against ground-truth text.

Usage:
    uv run python scripts/ab_rec_models.py IMAGE.png --truth truth.txt \
        [--candidates candidates.yaml]

Scoring per candidate model:
  - CER (character error rate) via difflib against ground truth
  - diacritic retention: fraction of truth's accented chars that appear
    correctly in the OCR output (the metric we actually care about)

The `baseline` candidate is always run: RapidOCR defaults, no overrides.

`candidates.yaml` format (fill with rec models you want to try — the exact
constructor kwargs depend on your installed rapidocr major version):

    - name: ppoocrv4-en-mobile
      kwargs:
        rec_model_type: en_ppocr_v4_mobile_rec   # rapidocr>=2 syntax
    - name: latin-custom
      kwargs:
        rec_model_path: /path/to/rec.onnx         # 1.x-style override
        rec_keys_path: /path/to/rec_keys.txt

Find candidate onnx rec models by searching Hugging Face for RapidOCR
model uploads (same maintainer hosts converted PP-OCR models), or export
them yourself from PaddleOCR model zoo.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import unicodedata
from pathlib import Path

ACCENTED = "áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def cer(truth: str, got: str) -> float:
    """Character error rate in [0, 1]; lower is better."""
    sm = difflib.SequenceMatcher(None, truth, got)
    if not truth:
        return 0.0 if not got else 1.0
    return 1.0 - sm.ratio()  # approximation adequate for ranking


def diacritic_retention(truth: str, got: str) -> tuple[int, int]:
    """(# accented truth chars matched, total accented truth chars)."""
    total = sum(1 for c in truth if c in ACCENTED)
    matched = 0
    got_pool = list(got)
    for c in truth:
        if c in ACCENTED and c in got_pool:
            got_pool.remove(c)  # consume; avoids double counting
            matched += 1
    return matched, total


def load_candidates(path: Path | None) -> list[tuple[str, dict]]:
    import yaml  # pip install pyyaml if needed

    cands: list[tuple[str, dict]] = [("baseline", {})]
    if path and path.exists():
        for entry in yaml.safe_load(path.read_text()) or []:
            cands.append((entry["name"], entry.get("kwargs", {})))
    return cands

def run_candidate(image: Path, kwargs: dict):
    try:
        from rapidocr import RapidOCR
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR

    # v3.x wants enum objects, not strings; resolve what we can
    engine = None
    if kwargs:
        try:
            from rapidocr import LangRec, ModelType, OCRVersion

            enum_map = {
                "Rec.ocr_version": OCRVersion,
                "Rec.model_type": ModelType,
                "Rec.lang_type": LangRec,
                "Det.ocr_version": OCRVersion,
            }
            resolved = {
                k: enum_map[k][v] if k in enum_map else v
                for k, v in kwargs.items()
            }
            engine = RapidOCR(params=resolved)
        except (ImportError, KeyError):
            engine = RapidOCR(params=kwargs)  # old-style, let it fail loudly
    else:
        engine = RapidOCR()

    raw = engine(str(image))
    if isinstance(raw, tuple) and len(raw) == 2:
        raw = raw[0]

    scored = []  # (y_min, text) so we sort by position, not alphabet
    if raw:
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                pts, text = item[0], str(item[1])
            elif isinstance(item, dict):
                pts, text = item["box"], str(item["text"])
            else:
                pts, text = getattr(item, "box_pts", []), str(getattr(item, "txt", ""))
            y = min(float(p[1]) for p in pts) if pts else 0.0
            scored.append((y, text))
    return "\n".join(t for _, t in sorted(scored))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--truth", type=Path, required=True,
                    help="UTF-8 text file with expected lines")
    ap.add_argument("--candidates", type=Path, default=Path("eval/candidates.yaml"))
    args = ap.parse_args()

    truth = _norm(args.truth.read_text(encoding="utf-8"))
    print(f"{'candidate':<24} {'CER':>6} {'diacritics':>12} {'mean_conf':>9}")
    print("-" * 56)

    from rich.table import Table
    from rich.console import Console

    table = Table(show_header=False, box=None)
    table.add_column("candidate", width=24)
    table.add_column("CER", justify="right")
    table.add_column("diacritics", justify="right")
    table.add_column("mean conf", justify="right")
    rows = []

    for name, kwargs in load_candidates(args.candidates):
        try:
            got = _norm(run_candidate(args.image, kwargs))
        except Exception as e:
            rows.append((name, f"FAIL: {str(e)[:40]}...", "-", "-"))
            continue
        c = cer(truth, got)
        m, t = diacritic_retention(truth, got)
        pct = f"{m}/{t}" if t else "n/a"
        rows.append((name, f"{c:.3f}", pct, "-"))

    for r in rows:
        table.add_row(*r)
    Console().print(table)
    # hint next step
    best = min((r for r in rows if r[1] not in ("-",) and "FAIL" not in r[1]),
               key=lambda r: float(r[1]), default=None)
    if best:
        print(f"\nbest: {best[0]} (CER {best[1]}, diacritics {best[2]})")


if __name__ == "__main__":
    main()

