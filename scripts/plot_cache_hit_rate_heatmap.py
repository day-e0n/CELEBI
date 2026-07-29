#!/usr/bin/env python3
"""Plot per-query cache hit rate as a heatmap (rows=stage, cols=query position).

Same underlying data as plot_cache_hit_rate_timeline.py, but shows the
per-query (non-cumulative) hit rate for each position so you can see exactly
which queries in the sequence hit vs missed, instead of only the running
average.

    hit_rate(query) = backed_provider_count / (backed_provider_count + auto_populate_count + auto_skip_count)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def to_float(value: object) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def series_hit_rate(query_summary: Path, series_name: str) -> list[tuple[str, float]]:
    rows = [r for r in read_rows(query_summary) if r.get("series") == series_name]
    rows.sort(key=lambda r: int(r["position"]))
    out = []
    for r in rows:
        bp = to_float(r["fixed_page_backed_provider_count"])
        pop = to_float(r["fixed_page_auto_populate_count"])
        skip = to_float(r["fixed_page_auto_skip_count"])
        denom = bp + pop + skip
        rate = bp / denom * 100.0 if denom else 0.0
        out.append((r["query"], rate))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        action="append",
        nargs=3,
        metavar=("LABEL", "QUERY_SUMMARY_CSV", "SERIES_NAME"),
        required=True,
        help="Repeatable. One heatmap row per stage.",
    )
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    stages = []
    for label, csv_path, series_name in args.stage:
        stages.append((label, series_hit_rate(Path(csv_path), series_name)))

    n = len(stages[0][1])
    row_labels = [label for label, _ in stages]
    col_labels = [str(i + 1) for i in range(n)]
    arr = np.array([[rate for _, rate in series] for _, series in stages], dtype=float)

    cmap = LinearSegmentedColormap.from_list(
        "soft_teal", ["#f7fcfd", "#e0f3f8", "#99d8c9", "#41ae76", "#005824"]
    )
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(max(14, n * 0.62), 1.4 + 1.7 * len(stages)))
    im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
    cbar.set_label("Per-query cache hit rate (%)", fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xticks(range(n))
    ax.set_xticklabels(col_labels, fontsize=13)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=19)
    ax.set_xlabel("Query position in workload (order differs by stage)", fontsize=20)

    for i, (_, series) in enumerate(stages):
        for j, (query, value) in enumerate(series):
            if not math.isfinite(value):
                continue
            color = "white" if value > 55 else "black"
            ax.text(j, i, f"{query}\n{value:.0f}%", ha="center", va="center", fontsize=11, color=color)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
