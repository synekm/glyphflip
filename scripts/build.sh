#!/usr/bin/env bash
# Build all three glyphflip linux editions: cloud, local, all.
#
# Usage:  ./scripts/build.sh
#         GLYPHFLIP_REBUILD_ENV=1 ./scripts/build.sh   (recreate build venv)
#
# Uses a dedicated build venv with CPU-ONLY torch (via pytorch's cpu
# index) — pip's default linux torch is CUDA-enabled and would freeze
# ~2.5GB of unused nvidia libs into local/all editions. opus-mt runs
# on CPU by design.
#
# Prereqs (Arch): sudo pacman -S binutils   # pyinstaller needs ldd
#
# Produces:
#   dist/glyphflip-<version>-linux-x64-cloud.zip   no torch
#   dist/glyphflip-<version>-linux-x64-local.zip   CPU torch
#   dist/glyphflip-<version>-linux-x64-all.zip     CPU torch + both legs
# Drop-in mt bundles are built separately: scripts/build_mt_bundle.sh

set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v ldd >/dev/null; then
    echo "error: ldd not found — on Arch: sudo pacman -S binutils" >&2
    exit 1
fi

BUILD_ENV=".build-venv"

# --- dedicated build venv: CPU-only torch keeps artifacts ~2GB lighter ---
if [ ! -d "${BUILD_ENV}" ] || [ "${GLYPHFLIP_REBUILD_ENV:-0}" = "1" ]; then
    echo "==> creating build venv (${BUILD_ENV}) with CPU-only torch"
    rm -rf "${BUILD_ENV}"
    uv venv "${BUILD_ENV}"
    uv pip install --python "${BUILD_ENV}/bin/python" \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        torch
    uv pip install --python "${BUILD_ENV}/bin/python" \
        -e ".[build,local]"
fi

# all uv invocations below resolve into the build venv, not .venv
export UV_PROJECT_ENVIRONMENT="$(pwd)/${BUILD_ENV}"

if ! uv run python -c "import PyInstaller" >/dev/null 2>&1; then
    echo "error: pyinstaller missing in build venv" >&2
    exit 1
fi

VERSION=$(uv run python -c "from glyphflip import __version__; print(__version__)")

for EDITION in cloud local all; do
    echo "==> building edition: ${EDITION}"
    rm -rf "build/${EDITION}" "dist/gf-${EDITION}"
    GF_EDITION=$EDITION uv run pyinstaller gf.spec --noconfirm \
        --distpath dist --workpath "build/${EDITION}"

    mv dist/gf "dist/gf-${EDITION}"
    cp LICENSE NOTICE.md "dist/gf-${EDITION}/"

    echo "==> smoke test: ${EDITION}"
    "./dist/gf-${EDITION}/gf" --version

    echo "==> zipping ${EDITION}"
    (cd dist && zip -r -q \
        "glyphflip-${VERSION}-linux-x64-${EDITION}.zip" "gf-${EDITION}")
done

echo "==> editions:"
ls -lh dist/*.zip

