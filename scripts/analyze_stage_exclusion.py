#!/usr/bin/env python3
# wdy start
"""Analyze stage-output exclusion reuse potential for TPC-H query-pair runs.

This is a byte-level/stage-level upper-bound analysis, not exact data-lineage
matching. It answers: if outputs from one stage kind such as JOIN are not kept
as cache candidates after Qi, how much of the stage-level candidate reuse volume
for Qj disappears?
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

STAGE_ORDER = [
    "SCAN",
    "FILTER",
    "PROJECTION",
    "JOIN",
    "AGGREGATE",
    "PARTITION",
    "CONCAT",
    "SORT",
    "LIMIT",
    "CTE",
    "RESULT",
    "OTHER",
]
DEFAULT_CACHE_STAGES = [
    "SCAN",
    "FILTER",
    "PROJECTION",
    "JOIN",
    "AGGREGATE",
    "PARTITION",
    "CONCAT",
    "SORT",
    "LIMIT",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Output root from run_tpch_pair_scan_audit.py")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <run-root>/stage_exclusion")
    parser.add_argument(
        "--cache-stages",
        default=",".join(DEFAULT_CACHE_STAGES),
        help="Comma-separated stage kinds considered cache candidates",
    )
    parser.add_argument(
        "--exclude-stages",
        default="NONE,SCAN,FILTER,PROJECTION,JOIN,AGGREGATE,PARTITION,CONCAT,SORT,LIMIT",
        help="Comma-separated exclusion cases to evaluate. NONE means keep all cache stages.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Do not generate PNG plots")
    return parser.parse_args()


def to_int(value: object, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(str(value)))
    except ValueError:
        return default


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(str(value))
    except ValueError:
        return default


def parse_stage_list(text: str) -> list[str]:
    return [item.strip().upper() for item in text.split(",") if item.strip()]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_stage_summary(path: Path) -> dict[tuple[str, str, str], dict[str, int]]:
    by_key: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"input_bytes": 0, "output_bytes": 0})
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["experiment"], row["query"], row["stage_kind"].upper())
            by_key[key]["input_bytes"] += to_int(row.get("input_bytes"))
            by_key[key]["output_bytes"] += to_int(row.get("output_bytes"))
    return by_key


def read_scan_pair_summary(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    by_pair: dict[tuple[str, str], dict[str, float]] = {}
    if not path.exists():
        return by_pair
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["previous_query"], row["second_query"])
            by_pair[key] = {
                "second_query_materialized_uncompressed_gb": to_float(
                    row.get("second_query_materialized_uncompressed_gb")
                ),
                "overlap_reload_uncompressed_gb": to_float(row.get("overlap_reload_uncompressed_gb")),
                "overlap_reload_ratio_of_second_load_uncompressed": to_float(
                    row.get("overlap_reload_ratio_of_second_load_uncompressed")
                ),
            }
    return by_pair


def gb(value: int | float) -> float:
    return float(value) / 1e9


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    manifest: list[dict[str, str]],
    stage_summary: dict[tuple[str, str, str], dict[str, int]],
    scan_summary: dict[tuple[str, str], dict[str, float]],
    cache_stages: list[str],
    exclude_stages: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []

    for item in manifest:
        qi = item["previous_query"]
        qj = item["second_query"]
        experiment = f"{qi}_then_{qj}"
        scan = scan_summary.get((qi, qj), {})

        prev_output_by_stage = {
            stage: stage_summary.get((experiment, qi, stage), {}).get("output_bytes", 0)
            for stage in cache_stages
        }
        second_input_by_stage = {
            stage: stage_summary.get((experiment, qj, stage), {}).get("input_bytes", 0)
            for stage in cache_stages
        }
        second_output_by_stage = {
            stage: stage_summary.get((experiment, qj, stage), {}).get("output_bytes", 0)
            for stage in cache_stages
        }

        base_upper_bound = sum(
            min(prev_output_by_stage[stage], second_input_by_stage[stage]) for stage in cache_stages
        )
        base_prev_candidate = sum(prev_output_by_stage.values())
        second_input_demand = sum(second_input_by_stage.values())

        for stage in cache_stages:
            stage_rows.append(
                {
                    "experiment": experiment,
                    "previous_query": qi,
                    "second_query": qj,
                    "stage_kind": stage,
                    "previous_output_gb": gb(prev_output_by_stage[stage]),
                    "second_input_gb": gb(second_input_by_stage[stage]),
                    "second_output_gb": gb(second_output_by_stage[stage]),
                    "stage_reuse_upper_bound_gb": gb(
                        min(prev_output_by_stage[stage], second_input_by_stage[stage])
                    ),
                }
            )

        for excluded in exclude_stages:
            kept_stages = cache_stages if excluded == "NONE" else [s for s in cache_stages if s != excluded]
            retained_prev_candidate = sum(prev_output_by_stage[stage] for stage in kept_stages)
            reuse_upper_bound = sum(
                min(prev_output_by_stage[stage], second_input_by_stage[stage]) for stage in kept_stages
            )
            lost_upper_bound = base_upper_bound - reuse_upper_bound
            excluded_prev_output = 0 if excluded == "NONE" else prev_output_by_stage.get(excluded, 0)
            excluded_second_input = 0 if excluded == "NONE" else second_input_by_stage.get(excluded, 0)

            rows.append(
                {
                    "experiment": experiment,
                    "previous_query": qi,
                    "second_query": qj,
                    "excluded_stage": excluded,
                    "cache_stages": " ".join(kept_stages),
                    "previous_candidate_output_gb": gb(base_prev_candidate),
                    "retained_previous_candidate_output_gb": gb(retained_prev_candidate),
                    "excluded_previous_output_gb": gb(excluded_prev_output),
                    "second_stage_input_demand_gb": gb(second_input_demand),
                    "excluded_second_input_gb": gb(excluded_second_input),
                    "stage_reuse_upper_bound_gb": gb(reuse_upper_bound),
                    "stage_reuse_upper_bound_loss_gb": gb(lost_upper_bound),
                    "stage_reuse_upper_bound_loss_ratio": (lost_upper_bound / base_upper_bound)
                    if base_upper_bound
                    else "",
                    "second_query_materialized_uncompressed_gb": scan.get(
                        "second_query_materialized_uncompressed_gb", ""
                    ),
                    "overlap_reload_uncompressed_gb": scan.get("overlap_reload_uncompressed_gb", ""),
                    "overlap_reload_ratio_of_second_load_uncompressed": scan.get(
                        "overlap_reload_ratio_of_second_load_uncompressed", ""
                    ),
                }
            )

    return rows, stage_rows


def maybe_plot(output_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: could not import plotting libraries: {exc}")
        return

    df = pd.DataFrame(rows)
    pairs = list(dict.fromkeys(df["experiment"].tolist()))
    excluded = [stage for stage in ["NONE"] + STAGE_ORDER if stage in set(df["excluded_stage"])]

    for value_col, filename, title, cbar_label in [
        (
            "stage_reuse_upper_bound_gb",
            "stage_exclusion_reuse_upper_bound_gb.png",
            "Stage-level reuse upper bound after excluding one stage",
            "Upper-bound GB",
        ),
        (
            "stage_reuse_upper_bound_loss_ratio",
            "stage_exclusion_loss_ratio.png",
            "Reuse upper-bound loss from excluding one stage",
            "Loss ratio",
        ),
    ]:
        pivot = df.pivot_table(index="experiment", columns="excluded_stage", values=value_col, aggfunc="first")
        pivot = pivot.reindex(index=pairs, columns=excluded)
        fig, ax = plt.subplots(figsize=(max(8, len(excluded) * 0.9), max(3.5, len(pairs) * 0.6)))
        values = pivot.to_numpy(dtype=float)
        im = ax.imshow(values, cmap="YlOrRd", aspect="auto")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticklabels(pivot.index)
        ax.set_title(title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if math.isfinite(value):
                    text = f"{value:.2f}" if "gb" in value_col else f"{value:.2f}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    run_root = args.run_root
    output_dir = args.output_dir or run_root / "stage_exclusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_root / "manifest.csv"
    stage_summary_path = run_root / "stage_summary" / "stage_audit_by_query_stage.csv"
    scan_summary_path = run_root / "summary" / "scan_audit_by_pair_second_query.csv"
    for path in (manifest_path, stage_summary_path):
        if not path.exists():
            raise SystemExit(f"missing required file: {path}")

    cache_stages = parse_stage_list(args.cache_stages)
    exclude_stages = parse_stage_list(args.exclude_stages)
    manifest = read_manifest(manifest_path)
    stage_summary = read_stage_summary(stage_summary_path)
    scan_summary = read_scan_pair_summary(scan_summary_path)
    rows, stage_rows = analyze(manifest, stage_summary, scan_summary, cache_stages, exclude_stages)

    write_csv(
        output_dir / "stage_exclusion_by_pair.csv",
        rows,
        [
            "experiment",
            "previous_query",
            "second_query",
            "excluded_stage",
            "cache_stages",
            "previous_candidate_output_gb",
            "retained_previous_candidate_output_gb",
            "excluded_previous_output_gb",
            "second_stage_input_demand_gb",
            "excluded_second_input_gb",
            "stage_reuse_upper_bound_gb",
            "stage_reuse_upper_bound_loss_gb",
            "stage_reuse_upper_bound_loss_ratio",
            "second_query_materialized_uncompressed_gb",
            "overlap_reload_uncompressed_gb",
            "overlap_reload_ratio_of_second_load_uncompressed",
        ],
    )
    write_csv(
        output_dir / "stage_overlap_components.csv",
        stage_rows,
        [
            "experiment",
            "previous_query",
            "second_query",
            "stage_kind",
            "previous_output_gb",
            "second_input_gb",
            "second_output_gb",
            "stage_reuse_upper_bound_gb",
        ],
    )

    if not args.no_plots:
        maybe_plot(output_dir, rows)

    print(f"wrote: {output_dir}")
    print("NOTE: this is a stage/byte upper-bound analysis, not exact lineage overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
