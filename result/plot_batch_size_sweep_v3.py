#!/usr/bin/env python3
"""ClickBench lazy batching on the current engine: total time vs number of batches.

The 41 queries are split at random into B batches (two splits, seeds 1 and 2); each batch keeps
the arrival order, is reordered on its own, and the page cache is told only the rest of the
current batch (order-aware eviction). Batches run back to back in one process, so the cache
carries over. Every split has its own baseline. O_DIRECT, prefetch off, executions 2-3.
Engine: late materialization on, scan no-history reservation 3x.

  N=41 (B=1): experiment/A_cb_{base,lru,best}_r{1,2}
  N=20-21 (B=2), N=10-11 (B=4), N=5-6 (B=8):
    experiment/NS3_cb_b{2,4,8}_s{1,2}_{base,lru,best}

  pixi run -e duckdb-python python result/plot_batch_size_sweep_v3.py
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

EXP = ROOT / "experiment"
OUT = Path(__file__).resolve().parent
# Match the supplied cumulative-query-time reference.
CONDS = [("Baseline", "base", "#4c78a8"),
         ("Baseline+page caching", "lru", "#59a14f"),
         ("Baseline+page caching+reordering (CELEBI)", "best", "#ffa514")]
RC = {"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.55,
      "pdf.fonttype": 42, "svg.fonttype": "none"}
BATCHES = [(1, "1"), (2, "2"), (4, "4")]


def runs(batches: int, cond: str) -> list[str]:
    if batches == 1:
        return [f"A_cb_{cond}_r{r}" for r in (1, 2)]
    return [f"NS3_cb_b{batches}_s{s}_{cond}" for s in (1, 2)]


def total(run: str) -> float:
    per: dict[int, float] = {}
    for line in open(EXP / run / "runner.log"):
        m = re.search(r"exec=(\d+)/\d+ .* done wall=([\d.]+)s", line)
        if m and int(m[1]) >= 2:
            per[int(m[1])] = per.get(int(m[1]), 0.0) + float(m[2])
    return sum(per.values()) / len(per)


def main() -> int:
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.4, 3.4))
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.21, top=0.79)
        width = 0.24
        upper = 0.0
        for gi, (b, label) in enumerate(BATCHES):
            for ci, (name, cond, color) in enumerate(CONDS):
                vals = [total(r) for r in runs(b, cond)]
                mean = statistics.mean(vals)
                x = gi + (ci - 1) * width
                ax.bar(x, mean, width * 0.86, color=color, edgecolor="#444444", linewidth=0.5,
                       zorder=3, label=name if gi == 0 else None)
                upper = max(upper, mean)
                caption = f"{mean:.1f}s"
                ax.annotate(caption, (x, mean), xytext=(0, 4),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=8, color="#333333", linespacing=1.3)
                print(f"{label:18} {name:42} {mean:6.2f}s")
        ax.set_xticks(range(len(BATCHES)), [b[1] for b in BATCHES], fontsize=8)
        ax.set_xlabel("# of batches", fontsize=10)
        ax.set_ylabel("Total query time (s)", fontsize=10)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=2.5, width=0.5, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#777777")
            spine.set_linewidth(0.55)
        ax.set_ylim(0, upper * 1.20)
        ax.set_xlim(-0.55, len(BATCHES) - 0.45)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.set_title("ClickBench", fontsize=11, pad=7)
        legend = fig.legend(fontsize=8, frameon=True, ncol=3, loc="upper center",
                            bbox_to_anchor=(0.54, 0.995), handlelength=1.4,
                            handletextpad=0.6, columnspacing=1.5, borderpad=0.55,
                            fancybox=False, edgecolor="#777777", framealpha=1)
        legend.get_frame().set_linewidth(0.55)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"batch_size_sweep_v3.{ext}", dpi=300, facecolor="white",
                        bbox_inches="tight")
        plt.close(fig)
    print(OUT / "batch_size_sweep_v3.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
