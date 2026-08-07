#!/usr/bin/env python3
"""qgen-varied real-execution scaling result: absolute total GPU operator work
time, Baseline vs CELEBI (page caching + reordering), across batch size N.
Reads experiment/expB_qgen_scaled_real_10to60/N{n}/comparison.csv (produced by
run_scaled_real_execution.py + build_operator_breakdown_comparison.py)."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_totals(n: int) -> tuple[float, float]:
    path = REPO_ROOT / "experiment" / "expB_qgen_scaled_real_10to60" / f"N{n}" / "comparison.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    total = next(r for r in rows if r["query"] == "TOTAL")
    return float(total["baseline_total_ms"]) / 1000.0, float(total["proposed_total_ms"]) / 1000.0


def main() -> int:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"

    ns = [10, 20, 30, 40, 50, 60]
    base_s, prop_s = zip(*[read_totals(n) for n in ns])

    color_base = "#2a78d6"
    color_prop = "#eda100"
    ink_primary = "#0b0b0b"

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(ns))
    width = 0.36
    bars_base = ax.bar(x - width / 2, base_s, width=width, color=color_base, label="Baseline")
    bars_prop = ax.bar(x + width / 2, prop_s, width=width, color=color_prop,
                        label="Baseline+page caching+reordering (CELEBI)")

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
    legend = ax.legend(frameon=True, fontsize=15, loc="upper left", labelcolor=ink_primary, edgecolor="black")
    legend.get_frame().set_linewidth(1.0)
    ax.set_ylim(0, max(base_s + prop_s) * 1.18)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    out_png = REPO_ROOT / "experiment" / "graph" / "expB_qgen_total_time_vs_n_celebi.png"
    out_pdf = REPO_ROOT / "experiment" / "graph" / "expB_qgen_total_time_vs_n_celebi.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
