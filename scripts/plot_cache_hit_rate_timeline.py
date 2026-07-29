#!/usr/bin/env python3
"""Plot cumulative fixed-page cache hit rate across query positions for one
or more pipeline stages (e.g. paging without reorder vs paging+reorder).

Hit rate per query position is defined over fixed-page-cache scan-provider
decisions:
    hit_rate = backed_provider_count / (backed_provider_count + auto_populate_count + auto_skip_count)
where:
  - backed_provider_count: scan served directly from an already-resident cached page (hit)
  - auto_populate_count:   page not resident, loaded and newly admitted into the cache (miss)
  - auto_skip_count:       page not resident, loaded but admission control rejected caching it (miss)

The cumulative version divides running totals of these three counters, so the
line shows how well the workload-so-far has been served from cache.
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def series_hit_counts(query_summary: Path, series_name: str) -> list[tuple[str, float, float, float]]:
    rows = [r for r in read_rows(query_summary) if r.get("series") == series_name]
    rows.sort(key=lambda r: int(r["position"]))
    return [
        (
            r["query"],
            to_float(r["fixed_page_backed_provider_count"]),
            to_float(r["fixed_page_auto_populate_count"]),
            to_float(r["fixed_page_auto_skip_count"]),
        )
        for r in rows
    ]


def cumulative_hit_rate(series: list[tuple[str, float, float, float]]) -> list[float]:
    out = []
    cbp = cpop = cskip = 0.0
    for _, bp, pop, skip in series:
        cbp += bp
        cpop += pop
        cskip += skip
        denom = cbp + cpop + cskip
        out.append(cbp / denom * 100.0 if denom else 0.0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        action="append",
        nargs=3,
        metavar=("LABEL", "QUERY_SUMMARY_CSV", "SERIES_NAME"),
        required=True,
        help="Repeatable. One cumulative-hit-rate line per stage.",
    )
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    colors = ["#4c78a8", "#59a14f", "#f2b134", "#e45756", "#9d6bb0"]

    stages = []
    for label, csv_path, series_name in args.stage:
        series = series_hit_counts(Path(csv_path), series_name)
        stages.append((label, series))

    n = len(stages[0][1])
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(max(11, n * 0.45), 7.2), constrained_layout=True)
    all_rates = []
    for idx, (label, series) in enumerate(stages):
        rate = cumulative_hit_rate(series)
        all_rates.append(rate)
        ax.plot(x, rate, color=colors[idx % len(colors)], linewidth=3.0, marker="o", markersize=5, label=f"{label} ({rate[-1]:.1f}%)")

    ax.set_ylabel("Cumulative cache hit rate (%)", fontsize=26)
    ax.set_xlabel("Query position in workload (order differs by stage)", fontsize=22)
    ax.set_ylim(0, max(v for rate in all_rates for v in rate) * 1.35)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=21)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=21, frameon=False)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=190, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
