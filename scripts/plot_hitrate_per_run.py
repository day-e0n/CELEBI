#!/usr/bin/env python3
"""Cold-run -> hot-run progression: mean cache hit rate (pooled across all 22
queries in a run) vs run index, for three stages (Baseline, Baseline+paging,
CELEBI). Input CSVs are produced by aggregate_hitrate_per_run.py.

Style matches plot_reorder_benefit_combined.py / plot_reorder_scaling.py:
black default spines, shared centered x-label, boxed top-center legend.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

COLORS = ["#4c78a8", "#59a14f", "#f2b134"]


def read_series(path: Path) -> list[tuple[int, float]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["run"]))
    return [(int(r["run"]), float(r["hit_rate_pct"])) for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        action="append",
        nargs=2,
        metavar=("LABEL", "CSV"),
        required=True,
        help="Repeatable: LABEL and the aggregate_hitrate_per_run.py output CSV for that stage.",
    )
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-pdf", type=Path, default=None)
    args = parser.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"

    ink_primary = "#0b0b0b"
    stages = [(label, read_series(Path(csv_path))) for label, csv_path in args.stage]

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for idx, (label, series) in enumerate(stages):
        runs = [r for r, _ in series]
        rates = [v for _, v in series]
        ax.plot(runs, rates, color=COLORS[idx % len(COLORS)], linewidth=2.6,
                 marker="o", markersize=8, label=label)
        ax.annotate(f"{rates[-1]:.1f}%", (runs[-1], rates[-1]),
                     textcoords="offset points", xytext=(-10, 10), ha="right",
                     fontsize=14, color=ink_primary)

    all_runs = sorted({r for _, series in stages for r, _ in series})
    ax.set_xticks(all_runs)
    ax.set_xlabel("run (cold -> hot)", fontsize=18, color=ink_primary)
    ax.set_ylabel("mean cache hit rate\nacross workload (%)", fontsize=18, color=ink_primary)
    ax.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=17, loc="upper left", labelcolor=ink_primary)

    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200)
    if args.out_pdf:
        fig.savefig(args.out_pdf)
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
