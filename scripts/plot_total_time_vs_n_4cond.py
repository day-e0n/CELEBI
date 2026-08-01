#!/usr/bin/env python3
"""Real-execution scaling result: absolute total GPU operator work time across
batch size N, 4 conditions: Baseline / Baseline+paging / CELEBI / CELEBI+fixed-start."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

NS = [10, 20, 30, 40, 50, 60]

CONDITIONS = [
    ("Baseline", REPO_ROOT / "experiment" / "expB_scaled_real", "baseline_total_ms", "#2a78d6"),
    ("Baseline+paging", REPO_ROOT / "experiment" / "expB_scaled_real_basepaging", "proposed_total_ms", "#eb6834"),
    ("CELEBI", REPO_ROOT / "experiment" / "expB_scaled_real_celebi", "proposed_total_ms", "#1baf7a"),
    ("CELEBI+fixed-start", REPO_ROOT / "experiment" / "expB_scaled_real", "proposed_total_ms", "#eda100"),
]


def read_total(root: Path, n: int, field: str) -> float:
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
        label: [read_total(root, n, field) for n in NS]
        for label, root, field, _color in CONDITIONS
    }

    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(NS))
    n_cond = len(CONDITIONS)
    width = 0.19
    offsets = [(-1.5 + i) * width for i in range(n_cond)]

    all_vals = []
    for (label, _root, _field, color), off in zip(CONDITIONS, offsets):
        vals = series[label]
        all_vals.extend(vals)
        bars = ax.bar(x + off, vals, width=width, color=color, label=label)
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}s", (rect.get_x() + rect.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4), fontsize=10,
                        color=ink_primary, ha="center", rotation=90 if n_cond > 3 else 0,
                        va="bottom")

    ax.set_xlabel("batch size N", fontsize=18, color=ink_primary)
    ax.set_ylabel("total GPU operator time (s)", fontsize=18, color=ink_primary)
    ax.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(NS)
    ax.legend(frameon=False, fontsize=14, loc="upper left", labelcolor=ink_primary, ncol=2)
    ax.set_ylim(0, max(all_vals) * 1.22)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_png = REPO_ROOT / "experiment" / "graph" / "expB_total_time_vs_n_4cond.png"
    out_pdf = REPO_ROOT / "experiment" / "graph" / "expB_total_time_vs_n_4cond.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
