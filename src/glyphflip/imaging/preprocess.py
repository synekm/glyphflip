"""Preprocessing helpers to improve OCR accuracy on small text.

PP-OCR detection struggles with tiny UI fonts (< ~20px tall). Screenshots
of app chrome (mail folders, menus, chat bubbles) are the classic case:
the whole image is small, so upscaling the whole frame is cheap and
helps both detection and recognition.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

#: images whose *smaller* side is below this get upscaled
_MIN_SIDE_PX = 800
#: never upscale more than this (diminishing returns + memory)
_MAX_FACTOR = 4.0


def maybe_upscale(arr: np.ndarray, min_side: int = _MIN_SIDE_PX) -> np.ndarray:
    """Return `arr` unchanged if already large enough, else Lanczos-upscaled.

    Factor is capped at _MAX_FACTOR; non-integer factors are fine (PIL
    handles fractional scaling).
    """
    h, w = arr.shape[:2]
    small_side = min(h, w)
    if small_side >= min_side:
        return arr
    factor = min(min_side / small_side, _MAX_FACTOR)
    img = Image.fromarray(arr)
    resized = img.resize(
        (round(w * factor), round(h * factor)), Image.LANCZOS
    )
    return np.asarray(resized, dtype=np.uint8)

