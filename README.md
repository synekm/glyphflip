# glyphflip

Feed it an image, get back the translated text.

`gf` is a CLI tool that OCRs text out of an image and translates it — fully local by default, cloud-assisted optionally. Point it at a screenshot of anything in one language, read the result in another.

```
$ gf translate screenshot.png --from cs --to en
Delivered
Sent
Drafts
Bulk
Archive
Spam
Basket
Starred
```

## Install

Prebuilt linux bundles (pick an edition, unzip, run):

| Edition | Size | Translation |
| - | - | - |
| `glyphflip-<ver>-linux-x64-cloud.zip` | \~200 MB | DeepL-compatible cloud API (key required) |
| `glyphflip-<ver>-linux-x64-local.zip` | \~500 MB | Local opus-mt models (offline capable) |
| `glyphflip-<ver>-linux-x64-all.zip` | \~500 MB | Local first, cloud fallback |

All editions bundle the OCR engine (RapidOCR / PP-OCRv6, Apache 2.0) and work out of the box for OCR. Translation models in local/all editions are fetched from Hugging Face on first use (\~300 MB per language pair, cached), or installed ahead of time as **drop-in bundles** (below).

From source (Python ≥ 3.10, Arch/example):

```sh
git clone <repo> && cd glyphflip
uv venv && source .venv/bin/activate
uv pip install -e ".[local,dev]"     # or "." for cloud-only
```

## Usage

```sh
gf translate <image> --from cs --to en        # translate (source optional with cloud autodetect)
gf translate <image> --to de                  # --from omitted: cloud leg autodetects
gf translate <image> --from cs --to en -v     # source/translation table
gf ocr <image>                                # OCR only
gf ocr <image> --show-boxes                   # + bboxes & confidence scores
gf models                                     # list installed drop-in model pairs
gf --version
```

Environment:

- `GF_CLOUD_KEY` — cloud translation API key (DeepL-compatible)
- `GF_MODEL_DIR` — directory containing a `models/` store of drop-in opus-mt weights (default: auto-detects `models/` next to the binary)
- `GF_HUB_OFFLINE=0` — force Hugging Face online lookups (offline by default; first-ever model download needs this)

## Drop-in translation models

Local editions ship without translation weights. Grab per-pair bundles and unzip them next to the `gf` binary (creates `models/`):

```sh
unzip glyphflip-mt-cs-en.zip -d /path/to/gf-directory/
gf models   # verify: opus-mt-cs-en now listed
```

Fully offline operation: install the pairs you need, then any `gf translate` runs without network.

## Development

```sh
uv pip install -e ".[local,dev]"
pytest -q               # fast suite (no models, seconds)
pytest -m slow -q       # + real model round-trips (downloads on first run)
./scripts/build.sh              # build all three linux editions
./scripts/build_mt_bundle.sh cs-en   # build a drop-in model bundle
```

Structure: `src/glyphflip/` with pluggable seams — `ocr/` (any OCR backend implementing the `OcrBackend` protocol; RapidOCR is the default), `translate/` (any `Translator`; opus-mt local + DeepL-compatible cloud), `pipeline.py` (routing: primary translator first, fallbacks in order). New OCR engines or translation providers are new files, not rewrites.

## Status / known issues

- Detection can occasionally fuse adjacent UI icons into text regions (e.g. a compose-button icon glued to a folder label), producing garbage-leading tokens like `VOdeslané` at high confidence. Noisy screenshots with dense icon rows are the trigger; tracked for a detection-tuning pass against a real-screenshot eval corpus.
- Translations of short UI fragments lack context (folder names may translate literally). Mitigation planned via per-domain glossaries.
- v1 local translation serves direct pairs via English only (X→en / en→X); other pairs need the cloud leg.

## License

Apache 2.0 — see `LICENSE`. Third-party components and model-weight attribution obligations are detailed in `NOTICE.md`. Notably: opus-mt weights are CC-BY 4.0 (attributed; not redistributed inside app bundles), PP-OCR models and RapidOCR are Apache 2.0.
