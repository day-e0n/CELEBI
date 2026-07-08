#!/usr/bin/env python3
"""Plot Figure 2: memory waste of whole-column caching."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiment" / "graph"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--column-pages", type=int, default=1024)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--hit-cost", type=float, default=0.05)
    parser.add_argument("--page-cache-management-cost", type=float, default=0.01)
    return parser.parse_args()


def lru_scan_sequence(
    hot_pages: int,
    cache_pages: int,
    sequence_length: int,
    *,
    hit_cost: float,
    miss_cost: float = 1.0,
    management_cost: float = 0.0,
) -> tuple[int, int, int, float]:
    cache: OrderedDict[int, None] = OrderedDict()
    hits = 0
    misses = 0
    evictions = 0
    latency = 0.0
    for _ in range(sequence_length):
        for page in range(hot_pages):
            if page in cache:
                hits += 1
                latency += hit_cost + management_cost
                cache.move_to_end(page)
            else:
                misses += 1
                latency += miss_cost + management_cost
                if cache_pages > 0:
                    cache[page] = None
                    if len(cache) > cache_pages:
                        cache.popitem(last=False)
                        evictions += 1
    return hits, misses, evictions, latency


def build_rows(
    column_pages: int,
    sequence_length: int,
    *,
    hit_cost: float,
    page_cache_management_cost: float,
) -> list[dict[str, object]]:
    access_ratios = [0.01, 0.05, 0.10, 0.25, 0.50]
    budget_ratios = [0.05, 0.10, 0.25]
    rows: list[dict[str, object]] = []

    for budget_ratio in budget_ratios:
        budget_pages = max(1, round(column_pages * budget_ratio))
        for access_ratio in access_ratios:
            hot_pages = max(1, round(column_pages * access_ratio))
            useful_pages = hot_pages

            page_hits, page_misses, page_evictions, page_latency = lru_scan_sequence(
                hot_pages,
                budget_pages,
                sequence_length,
                hit_cost=hit_cost,
                management_cost=page_cache_management_cost,
            )
            page_resident_pages = min(hot_pages, budget_pages)
            rows.append(
                {
                    "policy": "page_cache",
                    "budget_ratio": budget_ratio,
                    "access_ratio": access_ratio,
                    "column_pages": column_pages,
                    "budget_pages": budget_pages,
                    "hot_pages": hot_pages,
                    "admitted": 1,
                    "resident_ratio": page_resident_pages / column_pages,
                    "required_resident_ratio": hot_pages / column_pages,
                    "useful_ratio": page_resident_pages / column_pages,
                    "useful_resident_ratio": 1.0 if page_resident_pages > 0 else 0.0,
                    "cache_hits": page_hits,
                    "cache_misses": page_misses,
                    "eviction_count": page_evictions,
                    "admission_fail_count": 0,
                    "sequence_latency_norm_pages": page_latency,
                }
            )

            whole_admitted = budget_pages >= column_pages
            if whole_admitted:
                whole_hits, whole_misses, whole_evictions, whole_latency = lru_scan_sequence(
                    hot_pages,
                    column_pages,
                    sequence_length,
                    hit_cost=hit_cost,
                    management_cost=0.0,
                )
                resident_pages = column_pages
                useful_resident_ratio = useful_pages / resident_pages
                useful_cached_pages = useful_pages
                admission_fails = 0
            else:
                whole_hits = 0
                whole_misses = sequence_length * hot_pages
                whole_evictions = sequence_length
                whole_latency = float(whole_misses)
                resident_pages = 0
                useful_resident_ratio = 0.0
                useful_cached_pages = 0
                admission_fails = sequence_length

            rows.append(
                {
                    "policy": "whole_column_cache",
                    "budget_ratio": budget_ratio,
                    "access_ratio": access_ratio,
                    "column_pages": column_pages,
                    "budget_pages": budget_pages,
                    "hot_pages": hot_pages,
                    "admitted": 1 if whole_admitted else 0,
                    "resident_ratio": resident_pages / column_pages,
                    "required_resident_ratio": 1.0,
                    "useful_ratio": useful_cached_pages / column_pages,
                    "useful_resident_ratio": useful_resident_ratio,
                    "cache_hits": whole_hits,
                    "cache_misses": whole_misses,
                    "eviction_count": whole_evictions,
                    "admission_fail_count": admission_fails,
                    "sequence_latency_norm_pages": whole_latency,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def subset(rows: list[dict[str, object]], policy: str, budget_ratio: float) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row["policy"] == policy and abs(float(row["budget_ratio"]) - budget_ratio) < 1e-9
    ]


def plot(rows: list[dict[str, object]], output_path: Path) -> None:
    budgets = [0.05, 0.10, 0.25]
    colors = {
        "page_cache": "#1b9e77",
        "whole_column_cache": "#d95f02",
    }
    markers = {
        "page_cache": "o",
        "whole_column_cache": "s",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Figure 2: Whole-column cache wastes memory for sub-column reuse", fontsize=15)

    panels = [
        ("required_resident_ratio", "Required resident bytes / C"),
        ("useful_resident_ratio", "Useful / resident ratio"),
        ("eviction_count", "Eviction or admission-fail count"),
        ("sequence_latency_norm_pages", "Normalized sequence latency"),
    ]

    for ax, (field, ylabel) in zip(axes.ravel(), panels, strict=True):
        for budget in budgets:
            for policy in ("whole_column_cache", "page_cache"):
                data = subset(rows, policy, budget)
                data.sort(key=lambda row: float(row["access_ratio"]))
                xs = [float(row["access_ratio"]) * 100.0 for row in data]
                ys = [float(row[field]) for row in data]
                label = f"{policy.replace('_', ' ')} / budget={budget * 100:.0f}% C"
                ax.plot(
                    xs,
                    ys,
                    marker=markers[policy],
                    linewidth=1.8,
                    color=colors[policy],
                    alpha=0.45 + budget,
                    linestyle="-" if policy == "page_cache" else "--",
                    label=label,
                )
        if field == "required_resident_ratio":
            for budget in budgets:
                ax.axhline(budget, color="gray", linestyle=":", linewidth=1.0)
                ax.text(50.5, budget, f"{budget * 100:.0f}% budget", va="center", fontsize=8)
        ax.set_xlabel("Accessed page ratio of column (%)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(
        args.column_pages,
        args.sequence_length,
        hit_cost=args.hit_cost,
        page_cache_management_cost=args.page_cache_management_cost,
    )
    csv_path = args.output_dir / "figure2_whole_column_cache_waste.csv"
    png_path = args.output_dir / "figure2_whole_column_cache_waste.png"
    write_csv(csv_path, rows)
    plot(rows, png_path)
    print(f"wrote: {csv_path}")
    print(f"wrote: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
