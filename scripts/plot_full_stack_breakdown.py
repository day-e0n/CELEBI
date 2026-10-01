#!/usr/bin/env python3
"""Where a ClickBench run's time goes, from query start to query end.

Three stacked views of the same 43 queries, with and without the page cache.
Left: the task lifecycle, overlap removed, so the bars sum to the wall clock.
Middle: the operators inside Computing, also overlap removed.
Right: what the summed metric reports for the same two runs, which is larger than
wall clock because tasks run concurrently.

  pixi run -e duckdb-python python scripts/plot_full_stack_breakdown.py
"""
from __future__ import annotations

import collections
import importlib.util
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1] / "experiment"
RUNS = [("no cache", "WC_cb_base"), ("page cache", "WC_cb_cache")]
WANT_EXEC = 2


def union_s(items):
    merged = []
    for s, e in sorted(items):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged) / 1e9


def gather(run: str):
    telemetry = next((ROOT / run).glob("*/telemetry_data"))
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "P", here / "parse_quent_operator_breakdown.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    query_label, plan_query, pipeline_plan, pipeline_chain = P.load_telemetry(telemetry)

    tasks = collections.defaultdict(list)
    pipeline_of = {}
    for path in sorted(telemetry.rglob("task/*.ndjson")):
        with path.open() as handle:
            for line in handle:
                obj = json.loads(line)
                st = obj.get("data", {}).get("state")
                tasks[obj["id"]].append((obj["timestamp"], st))
                if isinstance(st, dict) and "Created" in st and isinstance(st["Created"], dict):
                    pipeline_of[obj["id"]] = st["Created"].get("pipeline_uuid", "")

    tspans = collections.defaultdict(list)
    ospans = collections.defaultdict(list)
    summed = 0.0
    for tid, series in tasks.items():
        pid = pipeline_of.get(tid)
        if not pid:
            continue
        label = P.resolve_query_label(pid, plan_query, pipeline_plan, query_label)
        parsed = P.parse_label(label) if label else None
        if not parsed or parsed[2] != WANT_EXEC:
            continue
        series.sort(key=lambda e: e[0])
        chain = pipeline_chain.get(pid, {})
        for i, (ts, st) in enumerate(series):
            if i + 1 >= len(series):
                continue
            span = (ts, series[i + 1][0])
            key = next(iter(st)) if isinstance(st, dict) and st else str(st)
            tspans[key].append(span)
            if key == "Computing" and isinstance(st["Computing"], dict):
                ospans[chain.get(st["Computing"].get("current_operator_id"),
                                 "UNKNOWN")].append(span)
                summed += (span[1] - span[0]) / 1e9
    return tspans, ospans, summed


def main() -> None:
    data = {label: gather(run) for label, run in RUNS}

    # Task states that matter, in lifecycle order; the rest are sub-millisecond.
    task_keys = ["Queued", "Reserving", "Computing"]
    op_keys = ["HASH_GROUP_BY", "GPU_SCAN", "PROJECTION", "MERGE_GROUP_BY", "TOP_N"]
    shades = ["#3d6b66", "#7fa6a1", "#a8c3bf", "#cdddda", "#e7eeed"]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8),
                             gridspec_kw=dict(width_ratios=[1, 1, 1], wspace=0.42))
    x = np.arange(len(RUNS))

    # --- panel 1: task lifecycle, overlap removed ----------------------
    ax = axes[0]
    bottom = np.zeros(len(RUNS))
    for key, shade in zip(task_keys, shades):
        vals = np.array([union_s(data[lab][0].get(key, [])) for lab, _ in RUNS])
        ax.bar(x, vals, 0.5, bottom=bottom, label=key, zorder=3,
               facecolor=shade, edgecolor="#2f3b3a", linewidth=0.6)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 4:
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if shade == shades[0] else "#2f3b3a")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _ in RUNS], fontsize=8)
    ax.legend(fontsize=6.5, loc="upper left", handlelength=1.0, labelspacing=0.3)
    ax.set_title("task states\n(overlap removed)", fontsize=8.5, pad=4)
    ps.finish(ax, "seconds")

    # --- panel 2: operators inside Computing ---------------------------
    ax = axes[1]
    bottom = np.zeros(len(RUNS))
    for key, shade in zip(op_keys, shades):
        vals = np.array([union_s(data[lab][1].get(key, [])) for lab, _ in RUNS])
        ax.bar(x, vals, 0.5, bottom=bottom, label=key.replace("_", " ").title(),
               zorder=3, facecolor=shade, edgecolor="#2f3b3a", linewidth=0.6)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 4:
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if shade == shades[0] else "#2f3b3a")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _ in RUNS], fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right", handlelength=1.0, labelspacing=0.3)
    ax.set_title("operators in Computing\n(overlap removed)", fontsize=8.5, pad=4)
    ps.finish(ax, None)

    # --- panel 3: the summed metric ------------------------------------
    ax = axes[2]
    bottom = np.zeros(len(RUNS))
    for key, shade in zip(op_keys, shades):
        vals = np.array([sum(e - s for s, e in data[lab][1].get(key, [])) / 1e9
                         for lab, _ in RUNS])
        ax.bar(x, vals, 0.5, bottom=bottom, zorder=3,
               facecolor=shade, edgecolor="#2f3b3a", linewidth=0.6)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 8:
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if shade == shades[0] else "#2f3b3a")
        bottom += vals
    for xi, v in enumerate(bottom):
        ax.text(xi, v + 3, f"{v:.0f}s", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _ in RUNS], fontsize=8)
    ax.set_ylim(0, max(bottom) * 1.18)
    ax.set_title("same operators, summed\n(what the old metric showed)",
                 fontsize=8.5, pad=4)
    ps.finish(ax, None)

    ps.save(fig, "full_stack_breakdown")


if __name__ == "__main__":
    main()
