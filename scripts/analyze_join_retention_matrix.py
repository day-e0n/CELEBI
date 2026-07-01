#!/usr/bin/env python3
# wdy start
"""Build all-query JOIN retention matrices from Sirius stage-audit summaries.

For each ordered pair Qi -> Qj:
  retention_cost(Qi,Qj) = JOIN output bytes produced by Qi
  reuse_loss(Qi,Qj)     = min(JOIN output bytes of Qi, JOIN input bytes of Qj)
  efficiency(Qi,Qj)     = reuse_loss / retention_cost

This is a stage-level byte upper bound. It does not prove semantic lineage
identity between two JOIN outputs/inputs.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

QUERIES = [f"q{i}" for i in range(1, 23)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-summary",
        type=Path,
        required=True,
        help="stage_audit_by_query_stage.csv from summarize_stage_audit.py",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def to_int(value: str | None) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def read_join_bytes(path: Path) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    join_input = {q: 0 for q in QUERIES}
    join_output = {q: 0 for q in QUERIES}
    join_events = {q: 0 for q in QUERIES}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            query = row.get("query", "")
            if query not in join_input:
                continue
            if row.get("stage_kind") != "JOIN":
                continue
            join_input[query] += to_int(row.get("input_bytes"))
            join_output[query] += to_int(row.get("output_bytes"))
            join_events[query] += to_int(row.get("events"))
    return join_input, join_output, join_events


def matrix_value(qi: str, qj: str, join_input: dict[str, int], join_output: dict[str, int], kind: str) -> float | None:
    if qi == qj:
        return None
    if join_output[qi] == 0 or join_input[qj] == 0:
        return None
    reuse_loss = min(join_output[qi], join_input[qj])
    if kind == "cost":
        return join_output[qi] / 1e9
    if kind == "loss":
        return reuse_loss / 1e9
    if kind == "efficiency":
        return reuse_loss / join_output[qi] if join_output[qi] else None
    raise ValueError(kind)


def write_matrix(path: Path, join_input: dict[str, int], join_output: dict[str, int], kind: str) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["previous_query"] + QUERIES)
        for qi in QUERIES:
            row: list[str] = [qi]
            for qj in QUERIES:
                value = matrix_value(qi, qj, join_input, join_output, kind)
                row.append("" if value is None else f"{value:.9f}")
            writer.writerow(row)


def write_long(path: Path, join_input: dict[str, int], join_output: dict[str, int]) -> None:
    with path.open("w", newline="") as f:
        fieldnames = [
            "previous_query",
            "second_query",
            "join_retention_cost_gb",
            "reuse_loss_gb",
            "retention_efficiency",
            "previous_join_output_gb",
            "second_join_input_gb",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qi in QUERIES:
            for qj in QUERIES:
                if qi == qj or join_output[qi] == 0 or join_input[qj] == 0:
                    writer.writerow(
                        {
                            "previous_query": qi,
                            "second_query": qj,
                            "join_retention_cost_gb": "",
                            "reuse_loss_gb": "",
                            "retention_efficiency": "",
                            "previous_join_output_gb": "" if join_output[qi] == 0 else join_output[qi] / 1e9,
                            "second_join_input_gb": "" if join_input[qj] == 0 else join_input[qj] / 1e9,
                        }
                    )
                    continue
                reuse_loss = min(join_output[qi], join_input[qj])
                writer.writerow(
                    {
                        "previous_query": qi,
                        "second_query": qj,
                        "join_retention_cost_gb": join_output[qi] / 1e9,
                        "reuse_loss_gb": reuse_loss / 1e9,
                        "retention_efficiency": reuse_loss / join_output[qi],
                        "previous_join_output_gb": join_output[qi] / 1e9,
                        "second_join_input_gb": join_input[qj] / 1e9,
                    }
                )


def write_query_summary(path: Path, join_input: dict[str, int], join_output: dict[str, int], join_events: dict[str, int]) -> None:
    with path.open("w", newline="") as f:
        fieldnames = ["query", "join_events", "join_input_gb", "join_output_gb", "has_join"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for q in QUERIES:
            writer.writerow(
                {
                    "query": q,
                    "join_events": join_events[q],
                    "join_input_gb": join_input[q] / 1e9 if join_input[q] else "",
                    "join_output_gb": join_output[q] / 1e9 if join_output[q] else "",
                    "has_join": int(join_events[q] > 0 and join_output[q] > 0),
                }
            )


def plot_heatmaps(output_dir: Path, join_input: dict[str, int], join_output: dict[str, int]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: could not import plotting libraries: {exc}")
        return

    specs = [
        ("cost", "join_retention_cost_gb_heatmap.png", "JOIN result retention cost", "GB", "YlOrRd"),
        ("loss", "join_reuse_loss_gb_heatmap.png", "Reuse loss if previous JOIN output is not retained", "GB", "YlOrRd"),
        ("efficiency", "join_retention_efficiency_heatmap.png", "JOIN retention efficiency", "reuse loss / retained bytes", "viridis"),
    ]
    for kind, filename, title, cbar_label, cmap in specs:
        values = []
        for qi in QUERIES:
            row = []
            for qj in QUERIES:
                value = matrix_value(qi, qj, join_input, join_output, kind)
                row.append(np.nan if value is None else value)
            values.append(row)
        arr = np.array(values, dtype=float)
        masked = np.ma.masked_invalid(arr)
        fig, ax = plt.subplots(figsize=(13, 11))
        im = ax.imshow(masked, cmap=cmap, aspect="auto")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        ax.set_xticks(range(len(QUERIES)))
        ax.set_yticks(range(len(QUERIES)))
        ax.set_xticklabels(QUERIES, rotation=90)
        ax.set_yticklabels(QUERIES)
        ax.set_xlabel("Second query (Qj)")
        ax.set_ylabel("Previous query (Qi)")
        ax.set_title(title)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                value = arr[i, j]
                if not math.isfinite(value):
                    continue
                text = f"{value:.1f}" if kind in ("cost", "loss") else f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=6, color="black")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    join_input, join_output, join_events = read_join_bytes(args.stage_summary)

    write_query_summary(args.output_dir / "join_query_summary.csv", join_input, join_output, join_events)
    write_long(args.output_dir / "join_retention_pair_long.csv", join_input, join_output)
    write_matrix(args.output_dir / "join_retention_cost_gb_matrix.csv", join_input, join_output, "cost")
    write_matrix(args.output_dir / "join_reuse_loss_gb_matrix.csv", join_input, join_output, "loss")
    write_matrix(args.output_dir / "join_retention_efficiency_matrix.csv", join_input, join_output, "efficiency")
    if not args.no_plots:
        plot_heatmaps(args.output_dir, join_input, join_output)

    nonzero = sum(1 for q in QUERIES if join_output[q] > 0)
    print(f"queries with JOIN output: {nonzero}/22")
    print(f"wrote: {args.output_dir}")
    print("NOTE: diagonal and queries without JOIN are blank in matrices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
