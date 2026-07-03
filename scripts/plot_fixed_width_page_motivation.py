#!/usr/bin/env python3
"""Create presentation-oriented motivation plots for fixed-width page reuse."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


# wdy start
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float | str] = {
                "pair": f'{row["previous_query"]}->{row["second_query"]}',
                "previous_query": row["previous_query"],
                "second_query": row["second_query"],
            }
            for key, value in row.items():
                if key in {"previous_query", "second_query"}:
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    rows.sort(key=lambda r: float(r["fixed_width_overlap_gb"]), reverse=True)
    return rows


def plot_motivation(rows: list[dict[str, float | str]], output_path: Path, top_n: int) -> None:
    rows = rows[:top_n]
    labels = [str(r["pair"]) for r in rows]
    load_ms = [float(r["second_load_ms_mean"]) for r in rows]
    compute_ms = [float(r["second_computation_ms_mean"]) for r in rows]
    overlap_gb = [float(r["fixed_width_overlap_gb"]) for r in rows]
    overlap_ratio = [float(r["fixed_width_overlap_ratio_of_second"]) for r in rows]
    load_fraction = [float(r["load_fraction_of_total"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.2, 1.0]})
    fig.suptitle("Motivation: SiriusDB repeatedly loads reusable fixed-width pages", fontsize=16)

    x = range(len(rows))
    axes[0].bar(x, load_ms, color="#d95f02", label="Load / materialization")
    axes[0].bar(x, compute_ms, bottom=load_ms, color="#7570b3", label="Computation")
    axes[0].set_title("Second-query latency breakdown")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].legend()
    for idx, (load, comp, frac) in enumerate(zip(load_ms, compute_ms, load_fraction)):
        axes[0].text(
            idx,
            load + comp + max(load_ms) * 0.03,
            f"{frac * 100:.0f}% load",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axes[1].bar(x, overlap_gb, color="#1b9e77", label="Reusable fixed-width pages")
    axes[1].set_title("Static reuse opportunity")
    axes[1].set_ylabel("Overlapped fixed-width page data (GB)")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    for idx, gb in enumerate(overlap_gb):
        axes[1].text(idx, gb + max(overlap_gb) * 0.03, f"{gb:.1f}GB", ha="center", fontsize=8)

    ax_ratio = axes[1].twinx()
    ax_ratio.plot(x, overlap_ratio, color="#e7298a", marker="o", linewidth=2, label="Overlap ratio")
    ax_ratio.set_ylim(0, 1.05)
    ax_ratio.set_ylabel("Overlap / second-query fixed-width footprint")

    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = ax_ratio.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_load_fraction_scatter(
    rows: list[dict[str, float | str]], output_path: Path, top_n: int
) -> None:
    rows = rows[:top_n]
    x = [float(r["fixed_width_overlap_gb"]) for r in rows]
    y = [float(r["load_fraction_of_total"]) * 100 for r in rows]
    sizes = [max(float(r["second_load_ms_mean"]) / 10.0, 40.0) for r in rows]
    labels = [str(r["pair"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=sizes, color="#1b9e77", alpha=0.75, edgecolor="black", linewidth=0.6)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Reusable fixed-width page data (GB)")
    ax.set_ylabel("Load fraction of second-query latency (%)")
    ax.set_title("High page overlap coincides with load-dominated execution")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.merged_csv)
    plot_motivation(rows, args.output_dir / "motivation_load_and_page_overlap.png", args.top_n)
    plot_load_fraction_scatter(
        rows, args.output_dir / "motivation_overlap_vs_load_fraction.png", args.top_n
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
