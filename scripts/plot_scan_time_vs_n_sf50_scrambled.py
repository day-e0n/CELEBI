#!/usr/bin/env python3
"""SF50 scrambled-order (deliberately worst-case adjacent-overlap arrival order)
scaling result: SCAN-only GPU operator work time across batch size N (join/
aggregate/filter/sort/other excluded), 3 conditions: Baseline / Baseline+page
caching / Baseline+page caching+query planning. 6GB fixed-page cache budget,
memory-pressure eviction enabled."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

NS = [10, 20, 30, 40, 50, 60]

# N=10/20/30/40/60 use the original 5-exec scrambled run; N=50 uses the 10-exec
# rerun (10x dirs) where available -- update as N=40/60 10-exec reruns land.
NEW_10X_NS = {50}

OLD_BASEPAGING = REPO_ROOT / "experiment" / "expB_scaled_real_sf50_scrambled_basepaging"
OLD_CELEBI = REPO_ROOT / "experiment" / "expB_scaled_real_sf50_scrambled_celebi"
NEW_BASEPAGING = REPO_ROOT / "experiment" / "expB_scaled_real_sf50_scrambled10x_basepaging"
NEW_CELEBI = REPO_ROOT / "experiment" / "expB_scaled_real_sf50_scrambled10x_celebi"

CONDITIONS = [
    ("Baseline", "basepaging", "baseline_scan", "#4C72B0"),
    ("Baseline+page caching", "basepaging", "proposed_scan", "#55A868"),
    ("Baseline+page caching+query planning", "celebi", "proposed_scan", "#E8A33D"),
]


def read_total(which: str, n: int, field: str) -> float:
    if which == "basepaging":
        root = NEW_BASEPAGING if n in NEW_10X_NS else OLD_BASEPAGING
    else:
        root = NEW_CELEBI if n in NEW_10X_NS else OLD_CELEBI
    path = root / f"N{n}" / "comparison.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    total = next(r for r in rows if r["query"] == "TOTAL")
    return float(total[field]) / 1000.0


def main() -> int:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"

    ink_primary = "#0b0b0b"

    series = {
        label: [read_total(which, n, field) for n in NS]
        for label, which, field, _color in CONDITIONS
    }

    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(NS))
    n_cond = len(CONDITIONS)
    width = 0.24
    offsets = [(i - (n_cond - 1) / 2) * width for i in range(n_cond)]

    all_vals = []
    for (label, _which, _field, color), off in zip(CONDITIONS, offsets):
        vals = series[label]
        all_vals.extend(vals)
        bars = ax.bar(x + off, vals, width=width, color=color, label=label,
                       edgecolor=ink_primary, linewidth=0.8)
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}s", (rect.get_x() + rect.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4), fontsize=14,
                        color=ink_primary, ha="center", rotation=90 if n_cond > 3 else 0,
                        va="bottom")

    ax.set_xlabel("batch size N", fontsize=22, color=ink_primary)
    ax.set_ylabel("scan GPU operator time (s)", fontsize=18, color=ink_primary)
    ax.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(NS)
    ax.set_ylim(0, max(all_vals) * 1.15)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(ink_primary)
        spine.set_linewidth(1.2)

    fig.tight_layout(rect=(0, 0, 1, 0.9))

    fig.legend(
        loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=n_cond,
        fontsize=16, frameon=True, edgecolor=ink_primary, labelcolor=ink_primary,
    )
    out_png = REPO_ROOT / "experiment" / "graph" / "expB_sf50_scrambled_scan_time_vs_n.png"
    out_pdf = REPO_ROOT / "experiment" / "graph" / "expB_sf50_scrambled_scan_time_vs_n.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
