#!/usr/bin/env python3
"""Real-execution scaling result: absolute total GPU operator work time
(baseline vs proposed) across batch size N."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_totals(n: int) -> tuple[float, float]:
    path = REPO_ROOT / "experiment" / "expB_scaled_real" / f"N{n}" / "comparison.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    total = next(r for r in rows if r["query"] == "TOTAL")
    return float(total["baseline_total_ms"]) / 1000.0, float(total["proposed_total_ms"]) / 1000.0


def main() -> int:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"

    ns = [10, 20, 30, 40, 50, 60]
    base_s, prop_s = zip(*[read_totals(n) for n in ns])

    color_base = "#2a78d6"
    color_prop = "#eb6834"
    ink_primary = "#0b0b0b"

    import numpy as np

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(ns))
    width = 0.36
    bars_base = ax.bar(x - width / 2, base_s, width=width, color=color_base, label="Baseline (no paging)")
    bars_prop = ax.bar(x + width / 2, prop_s, width=width, color=color_prop, label="Paging + reorder")

    for rect, b in zip(bars_base, base_s):
        ax.annotate(f"{b:.0f}s", (rect.get_x() + rect.get_width() / 2, b),
                    textcoords="offset points", xytext=(0, 6), fontsize=13, color=ink_primary, ha="center")
    for rect, p in zip(bars_prop, prop_s):
        ax.annotate(f"{p:.0f}s", (rect.get_x() + rect.get_width() / 2, p),
                    textcoords="offset points", xytext=(0, 6), fontsize=13, color=ink_primary, ha="center")

    ax.set_xlabel("batch size N", fontsize=18, color=ink_primary)
    ax.set_ylabel("total GPU operator time (s)", fontsize=18, color=ink_primary)
    ax.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(ns)
    ax.legend(frameon=False, fontsize=19, loc="upper left", labelcolor=ink_primary)
    ax.set_ylim(0, max(base_s + prop_s) * 1.18)

    fig.tight_layout()
    out_png = REPO_ROOT / "experiment" / "graph" / "expB_total_time_vs_n.png"
    out_pdf = REPO_ROOT / "experiment" / "graph" / "expB_total_time_vs_n.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
