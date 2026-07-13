#!/usr/bin/env python3
"""Plot fixed-page cache resident bytes over time for selected queries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def query_num(query: str) -> int:
    return int(query[1:]) if query.startswith("q") else 10_000


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


def plot(rows: list[dict[str, str]], queries: list[str], output: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 19,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    fig, axes = plt.subplots(len(queries), 1, figsize=(13.5, 3.8 * len(queries)), sharex=False)
    if len(queries) == 1:
        axes = [axes]

    for ax, query in zip(axes, queries):
        q_rows = [
            row
            for row in rows
            if row.get("query") == query
            and row.get("condition") == "paging"
            and row.get("event") == "page_directory"
        ]
        q_rows.sort(key=lambda row: to_float(row.get("relative_ms")) or 0.0)
        xs = [to_float(row.get("relative_ms")) / 1000.0 for row in q_rows]
        ys = [to_float(row.get("total_resident_gib") or row.get("resident_gib")) for row in q_rows]
        if xs:
            ax.step(xs, ys, where="post", color="#f58518", linewidth=2.8, label="fixed-page paging")
            ax.plot([0.0, max(xs)], [0.0, 0.0], color="#4c78a8", linewidth=2.3, label="baseline hot")
            ax.set_xlim(left=0.0, right=max(xs) * 1.03)
            ax.set_ylim(bottom=0.0)
            peak = max(ys)
            ax.text(
                0.98,
                0.82,
                f"peak resident: {peak:.2f} GiB",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=14,
            )
        else:
            ax.text(0.5, 0.5, "no paging residency events", transform=ax.transAxes, ha="center")
        ax.set_title(query, loc="left")
        ax.set_ylabel("Total cache resident (GiB)")
        ax.grid(axis="both", alpha=0.24)
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Time since process start (s)")
    fig.suptitle("Fixed-page total cache residency: baseline hot vs paging", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--queries", default="q6,q15,q20")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    queries = sorted(parse_csv_list(args.queries), key=query_num)
    plot(read_rows(args.events_csv), queries, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
