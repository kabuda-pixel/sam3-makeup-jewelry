"""Portable input paths and one binary-mask destination per source image."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def normalized_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def add_io_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=normalized_path,
                        help="Image directory, image file, or .txt image list.")
    parser.add_argument("--output-dir", required=True, type=normalized_path,
                        help="Directory for binary PNG masks only.")
    parser.add_argument("--checkpoint-path", required=True, type=normalized_path,
                        help="Local SAM3 checkpoint file.")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True,
                        help="Scan subdirectories (default: enabled).")


def iter_images(path: Path, recursive: bool = True) -> list[Path]:
    path = normalized_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if path.is_dir():
        candidates = path.rglob("*") if recursive else path.iterdir()
        images = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    elif path.suffix.lower() == ".txt":
        images = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = Path(os.path.expandvars(line.strip())).expanduser()
            images.append(normalized_path(item if item.is_absolute() else path.parent / item))
    else:
        images = [path]
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(f"Input image does not exist: {image}")
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension: {image}")
    images = list(dict.fromkeys(images))
    if not images:
        raise ValueError(f"No supported images found in: {path}")
    return images


def prepare_io(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    """Validate before loading models; preserve source names and subdirectories."""
    args.input = normalized_path(args.input)
    args.output_dir = normalized_path(args.output_dir)
    args.config = normalized_path(args.config)
    args.checkpoint_path = normalized_path(args.checkpoint_path)
    for label, path in (("Checkpoint", args.checkpoint_path), ("Config", args.config)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
    if args.input.is_dir() and args.output_dir.is_relative_to(args.input):
        raise ValueError("Output directory must be outside the input directory.")
    images = iter_images(args.input, args.recursive)
    if args.input.is_dir():
        base = args.input
    else:
        base = Path(os.path.commonpath([str(image.parent) for image in images]))
    jobs = []
    sources = {image.resolve() for image in images}
    destinations = set()
    for image in images:
        relative = image.relative_to(base)
        # Keep the source extension: 001.jpg and 001.png must have distinct masks.
        destination = args.output_dir / relative.parent / (relative.name + ".png")
        resolved = destination.resolve()
        if resolved in sources or resolved in destinations:
            raise ValueError(f"Output path collides with another file: {destination}")
        destinations.add(resolved)
        jobs.append((image, destination))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return jobs
