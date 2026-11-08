# NOTICE

This file contains attribution and license information for third-party
components used by glyphflip, as required by their respective licenses.
glyphflip itself is licensed under the Apache License, Version 2.0 —
see [LICENSE](LICENSE).

## Bundled components (shipped inside the glyphflip package / binary)

| Component | Upstream | License | Where it appears |
|---|---|---|---|
| PP-OCRv6 detection, recognition & classifier models | Baidu / PaddleOCR project, converted by RapidOCR (SWHL) | Apache-2.0 | OCR engine models |
| RapidOCR | RapidAI (SWHL and contributors) | Apache-2.0 | OCR runtime |
| onnxruntime | Microsoft | MIT | OCR inference runtime |
| OpenCV (headless build) | OpenCV team | Apache-2.0 | image ops (rapidocr dep) |
| Pillow | Jeffrey A. Clark (ACL) et al. | HPND / MIT-CMU | image loading & upscaling |
| NumPy | NumPy developers | BSD-3-Clause | arrays everywhere |
| Typer | Sebastián Ramírez | MIT | CLI |
| Rich | Will McGugan | MIT | terminal formatting |
| httpx | Encode Ltd | BSD-3-Clause | cloud translation client |

## Components fetched at runtime (NOT distributed with glyphflip)

Users of the local-translation feature download these separately on first
use, via the Hugging Face Hub. glyphflip does not redistribute them and
assumes no copyright over them:

| Component | Upstream | License | Note |
|---|---|---|---|
| opus-mt translation models (per language pair, e.g. opus-mt-cs-en) | Helsinki-NLP (University of Helsinki) | CC-BY 4.0 | attribution + license link required if *you* redistribute; fetching for internal use needs nothing beyond the license terms |
| HF Hub model files caching | Hugging Face | — | cache lives under the user's home dir |

Attribution for opus-mt (required if redistributed): Tiedemann, J. &
Thottingal, S., "OPUS-MT – Building open translation services for the
World", EAMT 2020 (https://aclanthology.org/2020.eamt-1.61).

## Cloud translation provider

The optional cloud translator targets the DeepL API. No DeepL SDK is
bundled — just our own HTTP calls. Usage requires the user's own API key
and is subject to DeepL's terms of service. glyphflip is not affiliated
with DeepL SE, and DeepL trademarks belong to their owners.

## Your obligations when redistributing glyphflip

1. Keep this NOTICE file and the LICENSE file with all copies.
2. If you redistribute the opus-mt model weights (e.g. pre-baking them
   into a fat bundle), include their CC-BY 4.0 attribution and a link to
   https://creativecommons.org/licenses/by/4.0/.
3. The Apache-2.0 components (PaddleOCR models, RapidOCR, OpenCV) carry
   over their license and notice obligations — reproducing this file
   satisfies the reasonable minimum.

