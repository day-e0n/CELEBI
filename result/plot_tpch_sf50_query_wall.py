#!/usr/bin/env python3
"""TPC-H SF50 baseline vs CELEBI using measured query wall time only.

Input: completed FULL_CACHE_PIN latency cases, execution 1 excluded, 2-3 averaged.
Output: paired per-query bars and a two-bar workload sum. No operator-work scaling.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result"
CONDITIONS = (("baseline", "Baseline", "#4c78a8"),
              ("reorder_future", "CELEBI", "#f2b134"))


def save(fig, stem):
    for ext in ("png", "svg", "pdf"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


def style(ax):
    ax.grid(axis="y", color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.6)


def main():
    means, totals, cases = [], {}, {}
    for condition, label, _ in CONDITIONS:
        case = ROOT / f"experiment/FULL_CACHE_PIN_20261007/runs/latency/th_r1/{condition}"
        metadata = json.loads((case / "metadata.json").read_text())
        assert metadata["status"] == "complete" and metadata["comparison_valid"], case
        assert metadata["environment"]["SIRIUS_CACHE_HIT_TRACE"] == "0", case
        assert metadata["future_mode"] == ("evict" if condition == "reorder_future" else "off"), case
        with (case / "queries.csv").open() as source:
            rows = list(csv.DictReader(source))
        assert len(rows) == 66 and {(int(r["query"]), int(r["execution"])) for r in rows} == {
            (q, ex) for q in range(1, 23) for ex in (1, 2, 3)}, case
        assert all(r["status"] == "ok" for r in rows), case
        for q in range(1, 23):
            values = [float(r["wall_s"]) for r in rows if int(r["query"]) == q
                      and int(r["execution"]) in (2, 3)]
            assert len(values) == 2 and all(math.isfinite(v) and v > 0 for v in values)
            means.append(dict(condition=condition, label=label, query=f"q{q}", samples=2,
                              query_wall_s=statistics.mean(values), query_wall_sd_s=statistics.stdev(values)))
        totals[condition] = sum(r["query_wall_s"] for r in means if r["condition"] == condition)
        assert math.isclose(totals[condition], metadata["steady_state_workload_wall_s"], abs_tol=1e-8)
        cases[condition] = dict(path=str(case), metadata=metadata)
    baseline, proposed = totals["baseline"], totals["reorder_future"]
    reduction = (1 - proposed / baseline) * 100
    with (OUT / "tpch_sf50_query_wall.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(means[0]))
        writer.writeheader()
        writer.writerows(means)
    (OUT / "tpch_sf50_query_wall_provenance.json").write_text(json.dumps(dict(
        metric="Per-query execute().fetchall() wall-clock elapsed time, including all operators",
        excludes="Label/hint calls, SET, reorder preparation, startup and execution 1 warmup",
        averaged_executions=[2, 3], independent_runs=1, cases=cases, workload_sums_s=totals,
        latency_reduction_pct=reduction, speedup=baseline / proposed), indent=2) + "\n")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.bbox": None}):
        fig, ax = plt.subplots(figsize=(10.5, 3.9))
        fig.subplots_adjust(left=0.075, right=0.98, bottom=0.17, top=0.79)
        x = np.arange(22)
        width = 0.35
        for offset, (condition, label, color) in zip((-0.5, 0.5), CONDITIONS):
            ys = [r["query_wall_s"] * 1000 for r in means if r["condition"] == condition]
            ax.bar(x + offset * width, ys, width=width, color=color,
                   edgecolor="#555555", linewidth=0.45, label=label, zorder=3)
        ax.set_xticks(x, [f"q{q}" for q in range(1, 23)])
        ax.set_xlim(-0.7, 21.7)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("TPC-H query")
        ax.set_ylabel("Query latency (ms)")
        ax.set_title("TPC-H SF50", fontsize=12, pad=8)
        style(ax)
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 0.98), frameon=True, edgecolor="#777777", fontsize=10)
        save(fig, "tpch_sf50_query_wall_per_query")

        fig, ax = plt.subplots(figsize=(4.3, 3.8))
        fig.subplots_adjust(left=0.18, right=0.96, bottom=0.17, top=0.85)
        values = [baseline, proposed]
        ax.bar([0, 1], values, width=0.58, color=[c[2] for c in CONDITIONS],
               edgecolor="#555555", linewidth=0.6, zorder=3)
        ax.set_xticks([0, 1], [c[1] for c in CONDITIONS])
        ax.set_ylim(0, 40)
        ax.set_ylabel("Total query time (s)")
        ax.set_title("TPC-H SF50 · 22 queries", fontsize=11, pad=9)
        ax.text(0, baseline + 1, f"{baseline:.2f}s", ha="center", va="bottom", fontsize=11)
        ax.text(1, proposed + 1, f"{proposed:.2f}s\n−{reduction:.1f}%", ha="center", va="bottom", fontsize=11)
        style(ax)
        save(fig, "tpch_sf50_query_wall_total")
    print(json.dumps(dict(totals_s=totals, reduction_pct=reduction, speedup=baseline / proposed)))


if __name__ == "__main__":
    main()
