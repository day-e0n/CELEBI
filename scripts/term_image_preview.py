#!/usr/bin/env python3
# wdy start
"""Render an image as ANSI true-color blocks in a terminal."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--width", type=int, default=100, help="terminal columns to use")
    return parser.parse_args()


def clamp_width(width: int) -> int:
    terminal_width = shutil.get_terminal_size((120, 40)).columns
    return max(10, min(width, terminal_width))


def main() -> int:
    args = parse_args()
    if not args.image.exists():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 1

    width = clamp_width(args.width)
    image = Image.open(args.image).convert("RGB")
    src_w, src_h = image.size
    height = max(1, int(src_h / src_w * width * 0.5))
    image = image.resize((width, height * 2))

    pixels = image.load()
    reset = "\x1b[0m"
    lines: list[str] = []
    for y in range(0, image.height, 2):
        parts: list[str] = []
        for x in range(image.width):
            tr, tg, tb = pixels[x, y]
            br, bg, bb = pixels[x, min(y + 1, image.height - 1)]
            parts.append(f"\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m▀")
        lines.append("".join(parts) + reset)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
