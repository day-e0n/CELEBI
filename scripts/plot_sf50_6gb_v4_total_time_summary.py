"""SF50/6GB fixed-page cache: total GPU operator time by caching strategy.

Reads summary/workload_execution_breakdown.csv from each of the 4 condition run
directories (final_sf50_6gb_v4_{baseline,paging,celebi,celebi_fixedstart}) and
plots one bar per condition: total_ms summed across all 8 executions (1 cold + 7
hot), in seconds.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "experiment" / "fixed_page_runs"

CONDITIONS = [
    ("Baseline", "final_sf50_6gb_v4_baseline", "#2a78d6"),
    ("Baseline\n+paging", "final_sf50_6gb_v4_paging", "#eb6834"),
    ("CELEBI", "final_sf50_6gb_v4_celebi", "#1baf7a"),
    ("CELEBI\n+fixed-start", "final_sf50_6gb_v4_celebi_fixedstart", "#eda100"),
]

MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
INK = "#0b0b0b"


def load_total_seconds(run_dir: Path) -> float:
    csv_path = run_dir / "summary" / "workload_execution_breakdown.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return sum(float(r["total_ms"]) for r in rows) / 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-png", type=Path, default=REPO_ROOT / "experiment" / "graph" / "sf50_6gb_v4_total_time_summary.png")
    args = parser.parse_args()

    labels, totals, colors = [], [], []
    for label, dirname, color in CONDITIONS:
        totals.append(load_total_seconds(RUNS_DIR / dirname))
        labels.append(label)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    x = range(len(labels))
    bars = ax.bar(x, totals, width=0.55, color=colors, zorder=3)

    for rect, val in zip(bars, totals):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + max(totals) * 0.012,
            f"{val:.1f}s",
            ha="center",
            va="bottom",
            fontsize=12,
            color=INK,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11, color=INK)
    ax.set_ylabel("Total GPU operator time (s)", fontsize=12, color=INK)
    ax.set_ylim(0, max(totals) * 1.15)

    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors=MUTED, labelsize=10)
    ax.tick_params(axis="x", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE_AXIS)

    ax.set_title(
        "SF50 · 6GB fixed-page cache budget · 3GB admission cap · 22 queries × 8 runs (1 cold + 7 hot)",
        fontsize=10,
        color=MUTED,
        pad=14,
    )

    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200)
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
