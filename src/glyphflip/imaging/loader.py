"""Image loading for OCR backends.

Single place where "Path | bytes" input becomes a numpy array.
Supports the formats PIL supports out of the box (PNG/JPEG/BMP/etc).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from glyphflip.ocr.base import ImageLoadError

logger = logging.getLogger(__name__)


def load_image(image: Path | bytes) -> np.ndarray:
    """Load an image into an RGB numpy array (H, W, 3), uint8.

    Args:
        image: filesystem path to an image, or raw encoded bytes.

    Raises:
        ImageLoadError: path doesn't exist or bytes aren't decodable.
    """
    source: Path | io.BytesIO
    if isinstance(image, Path):
        if not image.is_file():
            raise ImageLoadError(f"no such image file: {image}")
        source = image
        what = str(image)
    else:
        source = io.BytesIO(image)
        what = "image bytes"

    try:
        with Image.open(source) as im:
            rgb = im.convert("RGB")
    except Exception as e:  # PIL raises a zoo of exceptions; unify them
        raise ImageLoadError(f"failed to decode {what}: {e}") from e

    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # defensive: RGB conversion should guarantee this
        raise ImageLoadError(f"unexpected decoded shape: {arr.shape}")
    return arr

