#!/usr/bin/env python3
"""Per-query eligible cache hit rate within the COLD (first) execution only,
for two conditions (e.g. paging vs CELEBI), at SF50 budget=6GB/admission=3GB.

eligible_hit_rate excludes dynamic_filter_scan skips from the denominator
(matches plot_cache_hit_rate_eligible.py's definition).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from plot_cache_hit_rate_eligible import all_query_segments, segment_counts  # noqa: E402


def cold_run_hit_rates(log_dir: Path, query_order: list[str]) -> list[tuple[str, float, int, int]]:
    segments = all_query_segments(log_dir)
    n = len(query_order)
    cold = segments[:n]
    out = []
    for qnum, seg in zip(query_order, cold):
        backed, extra, _dyn = segment_counts(seg)
        denom = backed + extra
        rate = backed / denom * 100.0 if denom else 0.0
        out.append((qnum, rate, backed, denom))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", nargs=3,
                         metavar=("LABEL", "LOG_DIR", "QUERY_ORDER_CSV"),
                         required=True,
                         help="LABEL, .../workload/log_dir, comma-separated query order e.g. q8,q1,...")
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"

    ink_primary = "#0b0b0b"
    colors = ["#59a14f", "#f2b134"]

    stages = []
    for label, log_dir, order_csv in args.stage:
        order = order_csv.split(",")
        rates = cold_run_hit_rates(Path(log_dir), order)
        stages.append((label, rates))

    n_queries = len(stages[0][1])
    x = np.arange(1, n_queries + 1)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    for idx, (label, rates) in enumerate(stages):
        vals = [r for _, r, _, _ in rates]
        ax.plot(x, vals, color=colors[idx % len(colors)], linewidth=2.4,
                marker="o", markersize=6, label=label)

    # x tick labels: show query number for the FIRST stage's order (position axis
    # is shared, but each stage may visit a different query at a given position --
    # annotate both if they differ).
    labels = []
    for i in range(n_queries):
        qs = {label: rates[i][0] for label, rates in stages}
        if len(set(qs.values())) == 1:
            labels.append(next(iter(qs.values())))
        else:
            labels.append("/".join(qs.values()))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=10, color=ink_primary)
    ax.set_xlim(0.5, n_queries + 0.5)

    ax.set_xlabel("query (by position in the cold run)", fontsize=16, color=ink_primary, labelpad=10)
    ax.set_ylabel("eligible cache hit rate (%)\ncold run only", fontsize=16, color=ink_primary)
    ax.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, edgecolor=ink_primary, fontsize=14,
              loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=len(stages),
              labelcolor=ink_primary)

    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out_png}")

    if args.out_csv:
        with args.out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "position", "query", "hit_rate_pct", "backed", "denom"])
            for label, rates in stages:
                for pos, (qnum, rate, backed, denom) in enumerate(rates, start=1):
                    w.writerow([label, pos, qnum, round(rate, 2), backed, denom])
        print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
