#!/usr/bin/env python3
"""Overlay narrow-range vs widened-range qgen substitution parameters on the
same overlap_ratio-vs-N axes, to check whether the persistently high
overlap_ratio at large N is an artifact of TPC-H's spec-narrow substitution
ranges or a structural property of reusing 22 templates (shared join keys)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--narrow-exhaustive-csv", type=Path, required=True)
    parser.add_argument("--wide-exhaustive-csv", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-pdf", type=Path, default=None)
    args = parser.parse_args()

    narrow = read_rows(args.narrow_exhaustive_csv)
    wide = read_rows(args.wide_exhaustive_csv)

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"

    color_narrow = "#2a78d6"
    color_wide = "#eb6834"
    ink_primary = "#0b0b0b"

    n_narrow = [int(r["n"]) for r in narrow]
    q_narrow = [float(r["overlap_ratio"]) for r in narrow]
    n_wide = [int(r["n"]) for r in wide]
    q_wide = [float(r["overlap_ratio"]) for r in wide]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(n_narrow, q_narrow, color=color_narrow, linewidth=2.4, marker="o", markersize=7,
            label="narrow (spec-default qgen ranges, 10 streams)")
    ax.plot(n_wide, q_wide, color=color_wide, linewidth=2.4, marker="s", markersize=7,
            label="widened (2-3x spec ranges, 20 streams)")

    for x, y in zip(n_narrow, q_narrow):
        if x == n_narrow[-1]:
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(6, 8),
                        fontsize=11, color=color_narrow)
    for x, y in zip(n_wide, q_wide):
        if x == n_wide[-1]:
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(6, -14),
                        fontsize=11, color=color_wide)

    ax.set_xlabel("batch size N", fontsize=16, color=ink_primary)
    ax.set_ylabel("reorder quality (overlap_ratio, exhaustive)", fontsize=15, color=ink_primary)
    ax.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=13)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=12, loc="lower right", labelcolor=ink_primary)

    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200)
    if args.out_pdf:
        fig.savefig(args.out_pdf)
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
