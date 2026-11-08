#!/usr/bin/env bash
# Build drop-in opus-mt model bundles.
#
# Usage:  ./scripts/build_mt_bundle.sh cs-en en-cs de-en ...
#         ./scripts/build_mt_bundle.sh --all       (every supported pair)
#
# Produces: dist/mt/glyphflip-mt-<pair>.zip
#   layout inside:  models/opus-mt-<pair>/
#   install: unzip next to the gf binary (creates ./models/ next to it),
#            or anywhere + export GF_MODEL_DIR=/that/dir
#
# Each bundle is validated through glyphflip's own resolver before
# zipping, so what ships is guaranteed to be discovered by the app.

set -euo pipefail

cd "$(dirname "$0")/.."

# this script's whole job is downloading; never inherit offline defaults
unset HF_HUB_OFFLINE || true

if ! uv run python -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "error: huggingface_hub missing — uv pip install -e '.[local]'" >&2
    exit 1
fi

PAIRS=()
if [ "${1:-}" = "--all" ]; then
    mapfile -t PAIRS < <(uv run python -c "
from glyphflip.translate.opus_mt import _DIRECT_PAIRS
print('\n'.join(f'{s}-{t}' for (s, t) in sorted(_DIRECT_PAIRS)))")
else
    PAIRS=("$@")
fi

if [ "${#PAIRS[@]}" -eq 0 ]; then
    echo "usage: $0 <pair...|--all>   e.g. $0 cs-en" >&2
    exit 1
fi

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

mkdir -p dist/mt

for pair in "${PAIRS[@]}"; do
    repo="Helsinki-NLP/opus-mt-${pair}"
    out="dist/mt/glyphflip-mt-${pair}.zip"
    model_dir="$STAGE/models/opus-mt-${pair}"
    out_abs="$(pwd)/${out}"

    echo "==> ${pair}: downloading ${repo}"
    uv run python - "${pair%%-*}" "${pair##*-}" "$model_dir" <<'PYEOF'
import sys

from huggingface_hub import snapshot_download

source, target, dest = sys.argv[1:4]
snapshot_download(
    repo_id=f"Helsinki-NLP/opus-mt-{source}-{target}",
    local_dir=dest,
    allow_patterns=[
        "config.json", "pytorch_model.bin", "*.safetensors",
        "vocab.json", "source.spm", "target.spm",
        "generation_config.json", "tokenizer_config.json",
        "special_tokens_map.json",
    ],
)
print("downloaded to", dest)
PYEOF

    echo "==> ${pair}: validating via glyphflip resolver"
    uv run python - "${pair%%-*}" "${pair##*-}" "$STAGE" <<'PYEOF'
import os
import sys

source, target, stage = sys.argv[1:4]
os.environ["GF_MODEL_DIR"] = stage

from glyphflip.translate.model_store import resolve_model_path

p = resolve_model_path(source, target)
assert p, f"resolver could not find model in {stage}"
print("resolver OK:", p)
PYEOF

    echo "==> ${pair}: zipping"
    rm -f "$out"
    (cd "$STAGE" && zip -r -q "$out_abs" models)

    echo "==> ${pair}: $(du -h "$out" | cut -f1) -> $out"
done

echo "==> all bundles:"
ls -lh dist/mt/

