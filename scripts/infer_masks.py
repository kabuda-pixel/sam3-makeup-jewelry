#!/usr/bin/env python3
"""Choose makeup or jewelry inference from the input directory name."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam3_face_attributes.paths import add_io_arguments, normalized_path


def select_task(input_path: Path, task: str = "auto") -> str:
    if task != "auto":
        return task
    folder = input_path if input_path.is_dir() else input_path.parent
    name = folder.name.casefold()
    if name not in {"makeup", "jewelry"}:
        raise ValueError(
            f"Cannot infer task from folder '{folder.name}'. "
            "Use a makeup/jewelry input folder or pass --task makeup|jewelry."
        )
    return name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Process all input images and save only binary PNG masks.",
        epilog="Extra options are forwarded to infer_makeup.py or infer_jewelry.py.",
        allow_abbrev=False,
    )
    add_io_arguments(parser)
    parser.add_argument("--task", choices=("auto", "makeup", "jewelry"), default="auto")
    parser.add_argument("--sam3-repo", type=normalized_path,
                        help="SAM3 source root; omit when SAM3 is already installed.")
    args, extra = parser.parse_known_args(argv)
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    try:
        task = select_task(args.input, args.task)
    except ValueError as error:
        parser.error(str(error))
    if args.sam3_repo is not None:
        if not (args.sam3_repo / "sam3" / "model_builder.py").is_file():
            parser.error(f"Not a SAM3 source root: {args.sam3_repo}")
        sys.path.insert(0, str(args.sam3_repo))
    print(f"Task: {task}; input: {args.input}; masks: {args.output_dir}", flush=True)
    module = importlib.import_module(f"infer_{task}")
    module.main([
        "--input", str(args.input),
        "--output-dir", str(args.output_dir),
        "--checkpoint-path", str(args.checkpoint_path),
        "--recursive" if args.recursive else "--no-recursive",
        *extra,
    ])


if __name__ == "__main__":
    main()
