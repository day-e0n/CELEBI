#!/usr/bin/env python3
"""Plot hot vs paging stage breakdown from an already-filtered CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


STAGES = [
    ("scan", "SCAN", "#4c78a8"),
    ("join", "JOIN", "#54a24b"),
    ("aggregate", "AGG", "#e45756"),
    ("partition", "PARTITION", "#b279a2"),
    ("concat", "CONCAT", "#9d755d"),
    ("filter", "FILTER", "#f58518"),
    ("projection", "PROJ", "#72b7b2"),
    ("sort", "SORT", "#ff9da6"),
    ("result", "RESULT", "#bab0ac"),
    ("other", "OTHER", "#8cd17d"),
]


def to_float(value: object) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def query_num(query: str) -> int:
    return int(query[1:]) if query.startswith("q") else 10_000


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot(rows: list[dict[str, str]], output: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.size": 19,
            "axes.titlesize": 25,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 16,
        }
    )

    by_key = {(row["query"], row["series"]): row for row in rows}
    queries = sorted(
        {
            q
            for q, series in by_key
            if series == "hot" and (q, "paging") in by_key
        },
        key=query_num,
    )

    group_gap = 1.55
    x = [i * group_gap for i in range(len(queries))]
    width = 0.34
    pair_offset = 0.23

    fig, ax = plt.subplots(figsize=(26, 10.2))
    for series in ["hot", "paging"]:
        xs = [i + (-pair_offset if series == "hot" else pair_offset) for i in x]
        bottoms = [0.0] * len(queries)
        for key, label, color in STAGES:
            vals = [to_float(by_key.get((q, series), {}).get(f"{key}_work_ms")) for q in queries]
            ax.bar(
                xs,
                vals,
                width=width,
                bottom=bottoms,
                color=color,
                label=label if series == "hot" else None,
                edgecolor="white",
                linewidth=0.3,
            )
            bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels(queries, rotation=40, ha="right", fontweight="medium")
    ax.set_ylabel("Stage work (ms)")
    ax.set_title(title, pad=20)
    ax.grid(axis="y", alpha=0.23)
    ax.tick_params(axis="both", which="major", labelsize=20, width=1.2, length=6)
    ax.set_axisbelow(True)
    for xpos in x[:-1]:
        ax.axvline(xpos + group_gap / 2, color="#eeeeee", linewidth=0.8, zorder=0)

    handles = [
        Patch(facecolor="#9a9a9a", edgecolor="white", label="left bar = hot"),
        Patch(facecolor="#666666", edgecolor="white", label="right bar = paging"),
    ]
    handles.extend(Patch(facecolor=color, label=label) for _key, label, color in STAGES)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=True,
        columnspacing=1.6,
        handlelength=1.4,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.86))
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="TPC-H hot vs paging: stage-level work breakdown")
    args = parser.parse_args()

    plot(read_rows(args.input_csv), args.output, args.title)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
