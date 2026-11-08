# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for glyphflip's `gf` CLI — edition-parametric.
#
# GF_EDITION=cloud  -> no torch/transformers (small artifact, cloud-leg only)
# GF_EDITION=local  -> torch included, cloud code present but no key needed
# GF_EDITION=all    -> everything (same binary as local; keys optional at runtime)
#
# In all editions, opus-mt WEIGHTS are never bundled — users either let
# them fetch from HF on first use, or drop in a glyphflip-mt-<pair>.zip
# next to the binary (see model_store.py).
#
# Build:  ./scripts/build.sh   (loops all editions)

import importlib.util
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

EDITION = os.environ.get("GF_EDITION", "all")  # cloud | local | all
WANT_TORCH = EDITION in ("local", "all")

# --- rapidocr: bundled onnx models + dynamic imports ------------------------
spec_path = importlib.util.find_spec("rapidocr")
rapidocr_dir = Path(spec_path.origin).parent
models_dir = rapidocr_dir / "models"

datas = collect_data_files("rapidocr")
if models_dir.is_dir():
    datas.append((str(models_dir), "rapidocr/models"))

hiddenimports = collect_submodules("rapidocr")
hiddenimports += ["typer", "rich"]

excludes: list[str] = []
if not WANT_TORCH:
    # cloud edition: strip the entire local-translation leg
    excludes += [
        "torch", "transformers", "sacremoses",
        "transformers.models.marian",
        "transformers.models.marian.tokenization_marian",
    ]
else:
    hiddenimports += [
        "torch", "transformers",
        "transformers.models.marian",
        "transformers.models.marian.tokenization_marian",
        "sacremoses",
    ]

hiddenimports += [
    "onnxruntime",
]

a = Analysis(
    ["src/glyphflip/cli.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="gf",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # CLI tool; keep stdout/stderr semantics
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="gf",
)

