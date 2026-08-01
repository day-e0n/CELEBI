#!/usr/bin/env python3
"""Per-execution total_ms for baseline / baseline+paging / CELEBI at SF50,
cache budget=6GB, admission cap=3GB (22 canonical TPC-H queries, sequential
runs). Also annotates page_budget_evicted_pages so cold-only vs sustained
eviction is visible directly on the chart.

Style matches plot_hitrate_per_run.py / plot_reorder_benefit_combined.py:
black default spines, fontsize 18 labels, boxed legend.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#4c78a8", "#59a14f", "#f2b134"]


def read_series(path: Path) -> list[tuple[int, float, int]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["run"]))
    return [(int(r["run"]), float(r["total_ms"]), int(float(r["evicted_pages"]))) for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"

    ink_primary = "#0b0b0b"
    stages = [
        ("Baseline", REPO_ROOT / "experiment/graph/sf50_6gb_baseline_per_run.csv"),
        ("Baseline+paging", REPO_ROOT / "experiment/graph/sf50_6gb_paging_per_run.csv"),
        ("CELEBI (paging+reorder)", REPO_ROOT / "experiment/graph/sf50_6gb_celebi_per_run.csv"),
    ]
    series = [(label, read_series(path)) for label, path in stages]

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for idx, (label, s) in enumerate(series):
        runs = [r for r, _, _ in s]
        vals = [v / 1000.0 for _, v, _ in s]
        evs = [e for _, _, e in s]
        ax.plot(runs, vals, color=COLORS[idx % len(COLORS)], linewidth=2.6,
                marker="o", markersize=8, label=label)
        for r, v, e in zip(runs, vals, evs):
            if e > 0:
                ax.annotate(f"evict\n{e}", (r, v), textcoords="offset points",
                            xytext=(0, 12), ha="center", fontsize=10,
                            color=COLORS[idx % len(COLORS)])

    all_runs = sorted({r for _, s in series for r, _, _ in s})
    ax.set_xticks(all_runs)
    ax.set_xlabel("run (cold -> hot)", fontsize=18, color=ink_primary)
    ax.set_ylabel("total time (s)\nfor all 22 queries", fontsize=18, color=ink_primary)
    ax.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, edgecolor=ink_primary, fontsize=14,
              loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3,
              labelcolor=ink_primary)

    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
