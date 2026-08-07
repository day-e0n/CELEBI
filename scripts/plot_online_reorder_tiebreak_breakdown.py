#!/usr/bin/env python3
"""Online resident-aware reorder: per-operator (scan/join/aggregate/filter/sort/
other/total) % change vs baseline, across the four tiebreak variants.

Same sources as plot_online_reorder_tiebreak_comparison.py's TOTAL row."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

OPERATORS = ["scan", "join", "aggregate", "filter", "sort", "other", "total_ms"]
OPERATOR_LABELS = ["scan", "join", "aggregate", "filter", "sort", "other", "TOTAL"]

SERIES = [
    ("future_overlap\n(original)", "experiment/random22_breakdown_sf50_10x_online_resident/comparison_celebi_online.csv", "#2a78d6"),
    ("previous_overlap\n(O(1))", "experiment/random22_breakdown_sf50_10x_online_previous/comparison_celebi_online.csv", "#eb6834"),
    ("fewest columns\nfirst", "experiment/random22_breakdown_sf50_10x_online_smallest/comparison_celebi_online.csv", "#1baf7a"),
    ("most columns\nfirst", "experiment/random22_breakdown_sf50_10x_online_resident_v2/comparison_celebi_online.csv", "#eda100"),
]


def read_pct_changes(comparison_csv: Path) -> list[float]:
    with comparison_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    total = next(r for r in rows if r["query"] == "TOTAL")
    return [float(total[f"{op}_change_pct"]) for op in OPERATORS]


def main() -> int:
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"

    ink_primary = "#0b0b0b"
    ink_secondary = "#52514e"

    series_pcts = [read_pct_changes(REPO_ROOT / path) for _, path, _ in SERIES]

    n_ops = len(OPERATORS)
    n_series = len(SERIES)
    x = np.arange(n_ops)
    width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for i, (label, _, color) in enumerate(SERIES):
        offsets = x + (i - (n_series - 1) / 2) * width
        vals = series_pcts[i]
        bars = ax.bar(offsets, vals, width=width * 0.92, color=color, label=label.replace("\n", " "))
        for rect, v in zip(bars, vals):
            va = "bottom" if v >= 0 else "top"
            dy = 3 if v >= 0 else -3
            ax.annotate(f"{v:.1f}", (rect.get_x() + rect.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, dy), fontsize=7.5,
                        color=ink_secondary, ha="center", va=va, rotation=90)

    ax.axhline(0, color=ink_primary, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(OPERATOR_LABELS, fontsize=12, color=ink_primary)
    ax.set_ylabel("% change vs baseline (positive = faster)", fontsize=13, color=ink_primary)
    ax.set_title("Online resident-aware reorder: per-operator latency change by tiebreak rule",
                 fontsize=13, color=ink_primary)
    ax.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=11)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 3, ymax * 1.28)
    ax.legend(loc="upper left", fontsize=10, frameon=False, ncol=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle("SF50, 6GB budget, 22-query random arrival, 10 executions (1 warmup dropped)",
                 fontsize=11, color=ink_secondary, y=1.02)
    fig.tight_layout()

    out_png = REPO_ROOT / "experiment" / "graph" / "online_reorder_tiebreak_breakdown.png"
    out_pdf = REPO_ROOT / "experiment" / "graph" / "online_reorder_tiebreak_breakdown.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
