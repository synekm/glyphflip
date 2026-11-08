"""`gf` CLI entrypoint.

Typer + rich. Heavy engine imports happen inside factory functions so
`gf --help` stays instant and model-free. Every backend import is
guarded: on the cloud-only edition (no torch), missing local legs
degrade to warnings, never tracebacks.
"""

from __future__ import annotations

import logging
import os
import sys
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from glyphflip import __version__

app = typer.Typer(
    name="gf",
    help="Feed it an image, get back the translated text.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)

logger = logging.getLogger(__name__)


class Engine(str, Enum):
    RAPIDOCR = "rapidocr"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"gf {__version__}")
        raise typer.Exit()


def _configure_logging(verbose: bool) -> None:
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)
    # huggingface_hub / httpx chat at INFO and flood real output
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # HF telemetry/revision checks: trust the local cache by default.
    # Without this, every run makes ~8 HTTP calls to check for updates.
    # An explicit export wins (model downloads etc.).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback,
        help="Print version and exit.", is_eager=True,
    ),
) -> None:
    """glyphflip — image to translated text."""


def _frozen_helper_guard() -> None:
    """Inside a pyinstaller bundle, libraries that spawn helper child
    processes (e.g. multiprocessing's resource tracker) relaunch this
    binary with raw CPython flags: `gf -B -c "<code>"`. Typer rejects
    the flags; the intended behavior is executing the code in the
    frozen interpreter, whose sys.path the bootloader has already set.
    Detect that pattern, exec the code, exit before typer ever sees it.
    Frozen-only: dev/pip installs never hit this path."""
    try:
        i = sys.argv.index("-c")
    except ValueError:
        return
    code = sys.argv[i + 1] if len(sys.argv) > i + 1 else ""
    exec(compile(code, "<frozen-helper>", "exec"))
    sys.exit(0)


def _build_ocr_engine(engine: Engine, *, no_upscale: bool = False) -> object:
    """Import lazily; keeps --help and typo-free startup fast."""
    if engine is Engine.RAPIDOCR:
        from glyphflip.ocr.rapidocr_backend import RapidocrBackend

        return RapidocrBackend(upscale_min_side=None if no_upscale else 800)
    raise typer.BadParameter(f"unknown engine: {engine}")


def _build_translators(api_key: str | None) -> tuple[object, ...]:
    """Primary local translator, plus cloud fallback if configured.

    Both legs are optional: a missing one produces a warning, not a
    crash — this is what makes cloud/local/all editions work from the
    same code.
    """
    translators: list[object] = []

    from glyphflip.translate import opus_mt as _opus_mt

    if _opus_mt._TORCH_AVAILABLE:
        translators.append(_opus_mt.OpusMtTranslator())
    else:
        err_console.print(
            "[yellow]local translator not available in this edition[/]"
        )

    if api_key:
        from glyphflip.translate.cloud import CloudTranslator

        translators.append(CloudTranslator(api_key))

    if not translators:
        err_console.print(
            "[yellow]no translator available: no local edition and no "
            "cloud API key (set GF_CLOUD_KEY or pass --api-key)[/]"
        )
    return tuple(translators)


def _fail(msg: str, code: int = 1) -> None:
    err_console.print(f"[bold red]error:[/] {msg}")
    raise typer.Exit(code)


def _check_languages(target: str, source: str | None) -> None:
    from glyphflip.translate.base import TRANSLATION_LANGUAGES

    if target not in TRANSLATION_LANGUAGES:
        _fail(f"unsupported target language: {target!r}", code=2)
    if source is not None and source not in TRANSLATION_LANGUAGES:
        _fail(f"unsupported source language: {source!r}", code=2)


def _parse_image(path: Path) -> Path:
    if not path.is_file():
        _fail(f"no such file: {path}")
    return path


@app.command()
def ocr(
    image: Path = typer.Argument(..., help="Image file to read text from."),
    engine: Engine = typer.Option(
        Engine.RAPIDOCR, "--engine", help="OCR backend to use."
    ),
    min_confidence: float = typer.Option(
        0.4, "--min-confidence", help="Drop boxes below this confidence."
    ),
    show_boxes: bool = typer.Option(
        False, "--show-boxes", help="Also print bounding boxes and scores."
    ),
    no_upscale: bool = typer.Option(
        False, "--no-upscale", help="Disable small-image upscaling."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run OCR only — useful for grading engine quality on an image."""
    _configure_logging(verbose)
    img = _parse_image(image)

    from glyphflip.ocr.base import OcrOptions

    backend = _build_ocr_engine(engine, no_upscale=no_upscale)
    options = OcrOptions(min_confidence=min_confidence)

    try:
        result = backend.recognize(img, options)  # type: ignore[attr-defined]
    except Exception as e:
        _fail(f"OCR failed: {e}")

    ordered = sorted(result.boxes, key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))
    if show_boxes:
        table = Table(title="ocr result")
        table.add_column("#", justify="right", style="dim")
        table.add_column("text", overflow="fold")
        table.add_column("bbox", style="dim")
        table.add_column("conf", justify="right")
        for i, b in enumerate(ordered, 1):
            table.add_row(
                str(i), b.text,
                "[" + ", ".join(f"{v:.0f}" for v in b.bbox) + "]",
                "-" if b.confidence is None else f"{b.confidence:.2f}",
            )
        console.print(table)
    else:
        console.print(result.raw_text)

    conf = result.mean_confidence()
    if conf is not None:
        err_console.print(
            f"[dim]{len(ordered)} boxes, mean confidence {conf:.2f}[/]",
        )


@app.command()
def models() -> None:
    """List drop-in opus-mt models found next to the binary / in GF_MODEL_DIR."""
    _configure_logging(verbose=False)
    from glyphflip.translate.model_store import list_installed_pairs

    found = list_installed_pairs()
    if not found:
        console.print(
            "No drop-in models found. Unpack a glyphflip-mt-<pair>.zip next "
            "to the binary (creating models/), or set GF_MODEL_DIR."
        )
        raise typer.Exit()
    table = Table(title="installed opus-mt models")
    table.add_column("pair", style="bold")
    table.add_column("path", overflow="fold")
    for m in found:
        table.add_row(m["pair"], m["path"])
    console.print(table)


@app.command()
def translate(
    image: Path = typer.Argument(..., help="Image file to read text from."),
    to: str = typer.Option(..., "--to", help="Target language code, e.g. 'cs'."),
    from_: str = typer.Option(
        None, "--from", help="Source language code; detected if a cloud key is set.",
        metavar="LANG",
    ),
    engine: Engine = typer.Option(
        Engine.RAPIDOCR, "--engine", help="OCR backend to use."
    ),
    ocr_lang: str = typer.Option(
        "en", "--ocr-lang", help="Hint for OCR models (letter/script family)."
    ),
    no_upscale: bool = typer.Option(
        False, "--no-upscale", help="Disable small-image upscaling."
    ),
    api_key: str = typer.Option(
        None, "--api-key",
        help="Cloud translation API key (falls back to GF_CLOUD_KEY env).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Translate the text inside IMAGE into language TO."""
    _configure_logging(verbose)
    img = _parse_image(image)

    key = api_key or os.environ.get("GF_CLOUD_KEY")

    from glyphflip.ocr.base import OcrOptions
    from glyphflip.pipeline import GlyphFlipPipeline, PipelineError
    from glyphflip.translate.base import UnsupportedPairError

    target = to.lower()
    source = from_.lower() if from_ else None
    _check_languages(target, source)

    translators = _build_translators(key)
    if not translators:
        _fail("no translator available — see messages above", code=2)

    pipe = GlyphFlipPipeline(
        ocr_backend=_build_ocr_engine(engine, no_upscale=no_upscale),  # type: ignore[arg-type]
        primary_translator=translators[0],  # type: ignore[arg-type]
        fallback_translators=translators[1:],  # type: ignore[arg-type]
    )

    try:
        result = pipe.process(
            img,
            target,
            source_language=source,
            ocr_options=OcrOptions(language=ocr_lang),
        )
    except PipelineError as e:
        _fail(str(e))
    except UnsupportedPairError as e:
        _fail(str(e), code=2)

    if verbose:
        table = Table(title="ocr -> translation")
        table.add_column("#", justify="right", style="dim")
        table.add_column("source", overflow="fold")
        table.add_column("translation", overflow="fold")
        ocr_text = result.source_text.splitlines()
        trans = result.translation.translations
        for i, seg in enumerate(ocr_text):
            table.add_row(str(i + 1), seg, trans[i] if i < len(trans) else "")
        console.print(table)

    console.print(result.translated_text)


def main() -> None:
    """Console-script entrypoint (declared in pyproject)."""
    if getattr(sys, "frozen", False):
        _frozen_helper_guard()
    app()


if __name__ == "__main__":
    main()
