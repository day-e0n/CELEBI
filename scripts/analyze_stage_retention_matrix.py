#!/usr/bin/env python3
# wdy start
"""Build all-query operator-stage retention matrices from Sirius stage-audit summaries.

For each ordered pair Qi -> Qj and a selected stage S:
  retention_cost(Qi,Qj) = S output bytes produced by Qi
  reuse_loss(Qi,Qj)     = min(S output bytes of Qi, S input bytes of Qj)
  efficiency(Qi,Qj)     = reuse_loss / retention_cost

This is a stage-level byte upper bound. It does not prove semantic lineage
identity between two stage outputs/inputs.
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
    parser.add_argument("--stage-kind", default="FILTER", help="Stage kind, e.g. FILTER, JOIN, AGGREGATE")
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


def read_stage_bytes(path: Path, stage_kind: str) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    stage_kind = stage_kind.upper()
    stage_input = {q: 0 for q in QUERIES}
    stage_output = {q: 0 for q in QUERIES}
    stage_events = {q: 0 for q in QUERIES}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            query = row.get("query", "")
            if query not in stage_input:
                continue
            if row.get("stage_kind") != stage_kind:
                continue
            stage_input[query] += to_int(row.get("input_bytes"))
            stage_output[query] += to_int(row.get("output_bytes"))
            stage_events[query] += to_int(row.get("events"))
    return stage_input, stage_output, stage_events


def matrix_value(qi: str, qj: str, stage_input: dict[str, int], stage_output: dict[str, int], kind: str) -> float | None:
    if qi == qj:
        return None
    if stage_output[qi] == 0 or stage_input[qj] == 0:
        return None
    reuse_loss = min(stage_output[qi], stage_input[qj])
    if kind == "cost":
        return stage_output[qi] / 1e9
    if kind == "loss":
        return reuse_loss / 1e9
    if kind == "efficiency":
        return reuse_loss / stage_output[qi] if stage_output[qi] else None
    raise ValueError(kind)


def write_matrix(path: Path, stage_input: dict[str, int], stage_output: dict[str, int], kind: str) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["previous_query"] + QUERIES)
        for qi in QUERIES:
            row: list[str] = [qi]
            for qj in QUERIES:
                value = matrix_value(qi, qj, stage_input, stage_output, kind)
                row.append("" if value is None else f"{value:.9f}")
            writer.writerow(row)


def write_long(path: Path, stage_input: dict[str, int], stage_output: dict[str, int], stage_kind: str) -> None:
    prefix = stage_kind.lower()
    with path.open("w", newline="") as f:
        fieldnames = [
            "previous_query",
            "second_query",
            f"{prefix}_retention_cost_gb",
            "reuse_loss_gb",
            "retention_efficiency",
            f"previous_{prefix}_output_gb",
            f"second_{prefix}_input_gb",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qi in QUERIES:
            for qj in QUERIES:
                base = {
                    "previous_query": qi,
                    "second_query": qj,
                    f"previous_{prefix}_output_gb": "" if stage_output[qi] == 0 else stage_output[qi] / 1e9,
                    f"second_{prefix}_input_gb": "" if stage_input[qj] == 0 else stage_input[qj] / 1e9,
                }
                if qi == qj or stage_output[qi] == 0 or stage_input[qj] == 0:
                    base.update(
                        {
                            f"{prefix}_retention_cost_gb": "",
                            "reuse_loss_gb": "",
                            "retention_efficiency": "",
                        }
                    )
                    writer.writerow(base)
                    continue
                reuse_loss = min(stage_output[qi], stage_input[qj])
                base.update(
                    {
                        f"{prefix}_retention_cost_gb": stage_output[qi] / 1e9,
                        "reuse_loss_gb": reuse_loss / 1e9,
                        "retention_efficiency": reuse_loss / stage_output[qi],
                    }
                )
                writer.writerow(base)


def write_query_summary(path: Path, stage_input: dict[str, int], stage_output: dict[str, int], stage_events: dict[str, int], stage_kind: str) -> None:
    prefix = stage_kind.lower()
    with path.open("w", newline="") as f:
        fieldnames = ["query", f"{prefix}_events", f"{prefix}_input_gb", f"{prefix}_output_gb", f"has_{prefix}"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for q in QUERIES:
            writer.writerow(
                {
                    "query": q,
                    f"{prefix}_events": stage_events[q],
                    f"{prefix}_input_gb": stage_input[q] / 1e9 if stage_input[q] else "",
                    f"{prefix}_output_gb": stage_output[q] / 1e9 if stage_output[q] else "",
                    f"has_{prefix}": int(stage_events[q] > 0 and stage_output[q] > 0),
                }
            )


def plot_heatmaps(output_dir: Path, stage_input: dict[str, int], stage_output: dict[str, int], stage_kind: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: could not import plotting libraries: {exc}")
        return

    label = stage_kind.upper()
    prefix = stage_kind.lower()
    specs = [
        ("cost", f"{prefix}_retention_cost_gb_heatmap.png", f"{label} result retention cost", "GB", "YlOrRd"),
        ("loss", f"{prefix}_reuse_loss_gb_heatmap.png", f"Potential reuse loss if previous {label} output is not retained", "GB", "YlOrRd"),
        ("efficiency", f"{prefix}_retention_efficiency_heatmap.png", f"{label} retention efficiency", "reuse loss / retained bytes", "viridis"),
    ]
    for kind, filename, title, cbar_label, cmap in specs:
        values = []
        for qi in QUERIES:
            row = []
            for qj in QUERIES:
                value = matrix_value(qi, qj, stage_input, stage_output, kind)
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
    stage_kind = args.stage_kind.upper()
    prefix = stage_kind.lower()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_input, stage_output, stage_events = read_stage_bytes(args.stage_summary, stage_kind)

    write_query_summary(args.output_dir / f"{prefix}_query_summary.csv", stage_input, stage_output, stage_events, stage_kind)
    write_long(args.output_dir / f"{prefix}_retention_pair_long.csv", stage_input, stage_output, stage_kind)
    write_matrix(args.output_dir / f"{prefix}_retention_cost_gb_matrix.csv", stage_input, stage_output, "cost")
    write_matrix(args.output_dir / f"{prefix}_reuse_loss_gb_matrix.csv", stage_input, stage_output, "loss")
    write_matrix(args.output_dir / f"{prefix}_retention_efficiency_matrix.csv", stage_input, stage_output, "efficiency")
    if not args.no_plots:
        plot_heatmaps(args.output_dir, stage_input, stage_output, stage_kind)

    nonzero = sum(1 for q in QUERIES if stage_output[q] > 0)
    print(f"queries with {stage_kind} output: {nonzero}/22")
    print(f"wrote: {args.output_dir}")
    print("NOTE: diagonal and queries without selected stage are blank in matrices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
