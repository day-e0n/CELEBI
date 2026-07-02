#!/usr/bin/env python3
"""Merge fixed-width page overlap potential with measured Sirius latency."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


# wdy start
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap-csv", required=True, type=Path)
    parser.add_argument("--latency-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-repeats", type=int, default=2)
    return parser.parse_args()


def read_overlap(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    rows: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["previous_query"], row["second_query"])
            rows[key] = {
                "fixed_width_overlap_gb": float(row["fixed_width_overlap_gb"]),
                "fixed_width_overlap_ratio_of_second": float(
                    row["fixed_width_page_overlap_ratio_of_second_query"]
                ),
                "fixed_width_overlap_pages": float(row["fixed_width_overlap_pages"]),
                "second_fixed_width_footprint_gb": float(
                    row["second_query_fixed_width_page_footprint_bytes"]
                )
                / 1e9,
                "previous_fixed_width_footprint_gb": float(
                    row["previous_query_fixed_width_page_footprint_bytes"]
                )
                / 1e9,
                "variable_width_second_footprint_gb": float(
                    row["variable_width_second_query_footprint_bytes"]
                )
                / 1e9,
            }
    return rows


def read_latency(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_merged(rows: list[dict[str, str]], output_path: Path) -> list[dict[str, float | str]]:
    fieldnames = [
        "previous_query",
        "second_query",
        "repeats_completed",
        "fixed_width_overlap_gb",
        "fixed_width_overlap_ratio_of_second",
        "fixed_width_overlap_pages",
        "previous_fixed_width_footprint_gb",
        "second_fixed_width_footprint_gb",
        "variable_width_second_footprint_gb",
        "second_total_ms_mean",
        "second_load_ms_mean",
        "second_computation_ms_mean",
        "load_fraction_of_total",
        "overlap_gb_per_second_load_ms",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_bars(rows: list[dict[str, float | str]], output_path: Path) -> None:
    rows = sorted(rows, key=lambda r: float(r["fixed_width_overlap_gb"]), reverse=True)
    labels = [f'{r["previous_query"]}->{r["second_query"]}' for r in rows]
    load = [float(r["second_load_ms_mean"]) for r in rows]
    comp = [float(r["second_computation_ms_mean"]) for r in rows]
    overlap = [float(r["fixed_width_overlap_gb"]) for r in rows]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = range(len(rows))
    ax.bar(x, load, label="Load latency")
    ax.bar(x, comp, bottom=load, label="Computation latency")
    ax.set_ylabel("Second query latency (ms)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(list(x), overlap, color="black", marker="o", linewidth=1.5, label="Page overlap GB")
    ax2.set_ylabel("Fixed-width page overlap (GB)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scatter(rows: list[dict[str, float | str]], output_path: Path) -> None:
    x = [float(r["fixed_width_overlap_gb"]) for r in rows]
    y = [float(r["second_load_ms_mean"]) for r in rows]
    labels = [f'{r["previous_query"]}->{r["second_query"]}' for r in rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=70)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("Fixed-width page overlap (GB)")
    ax.set_ylabel("Baseline second-query load latency (ms)")
    ax.set_title("Static page reuse potential vs measured load latency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlap = read_overlap(args.overlap_csv)
    merged: list[dict[str, float | str]] = []
    for row in read_latency(args.latency_csv):
        repeats = int(row["repeats_completed"])
        if repeats < args.min_repeats:
            continue
        key = (row["previous_query"], row["second_query"])
        if key not in overlap:
            continue
        total_ms = float(row["second_total_ms_mean"])
        load_ms = float(row["second_load_ms_mean"])
        comp_ms = float(row["second_computation_ms_mean"])
        overlap_gb = overlap[key]["fixed_width_overlap_gb"]
        merged.append(
            {
                "previous_query": key[0],
                "second_query": key[1],
                "repeats_completed": repeats,
                **overlap[key],
                "second_total_ms_mean": total_ms,
                "second_load_ms_mean": load_ms,
                "second_computation_ms_mean": comp_ms,
                "load_fraction_of_total": load_ms / total_ms if total_ms else 0.0,
                "overlap_gb_per_second_load_ms": overlap_gb / load_ms if load_ms else 0.0,
            }
        )

    merged.sort(key=lambda r: float(r["fixed_width_overlap_gb"]), reverse=True)
    write_merged(merged, args.output_dir / "fixed_width_page_motivation_merged.csv")
    plot_bars(merged, args.output_dir / "fixed_width_page_overlap_vs_latency.png")
    plot_scatter(merged, args.output_dir / "fixed_width_page_overlap_vs_load_scatter.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
