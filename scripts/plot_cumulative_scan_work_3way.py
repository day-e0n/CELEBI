#!/usr/bin/env python3
"""Cumulative scan/materialize work (s) by query position, for 3 stages
(baseline / paging / CELEBI), each read from its own query_summary_breakdown.csv.
Matches plot_cache_hit_rate_eligible.py's companion-panel style.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def to_float(value: object) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def cumulative_work_s(query_summary: Path, series_name: str) -> list[float]:
    with query_summary.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("series") == series_name]
    rows.sort(key=lambda r: int(r["position"]))
    out = []
    total = 0.0
    for r in rows:
        total += to_float(r.get("scan_materialize_work_ms"))
        out.append(total / 1000.0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", nargs=3,
                         metavar=("LABEL", "QUERY_SUMMARY_CSV", "SERIES_NAME"),
                         required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    ink_primary = "#0b0b0b"
    colors = ["#4c78a8", "#59a14f", "#f2b134"]

    stages = []
    for label, csv_path, series_name in args.stage:
        vals = cumulative_work_s(Path(csv_path), series_name)
        stages.append((label, vals))

    n = len(stages[0][1])
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    baseline_final = stages[0][1][-1]
    for idx, (label, vals) in enumerate(stages):
        ax.plot(x, vals, color=colors[idx % len(colors)], linewidth=2.6,
                marker="o", markersize=5, label=label)
        final = vals[-1]
        pct = (baseline_final - final) / baseline_final * 100.0 if baseline_final else 0.0
        text = f"{final:.1f}s" if idx == 0 else f"{final:.1f}s ({pct:+.1f}%)"
        ax.annotate(text, (x[-1], final), textcoords="offset points",
                    xytext=(8, 0), ha="left", va="center", fontsize=13, color=ink_primary)

    ax.set_xlabel("Query position in workload (order differs by stage)", fontsize=16, color=ink_primary)
    ax.set_ylabel("Cumulative scan/materialize\nwork (s)", fontsize=16, color=ink_primary)
    ax.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=13)
    ax.set_xlim(0, n + 3)
    ax.legend(frameon=True, edgecolor=ink_primary, fontsize=14,
              loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3,
              labelcolor=ink_primary)

    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
