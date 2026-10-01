#!/usr/bin/env python3
"""How many tasks are computing at each instant, and what they are computing.

The height of the band is the number of tasks in Computing at that moment -- the
overlap itself, rather than 500 individual rows. Colour is the operator they are
running. The queries that dominate are named above the band.

Area under the band = summed GPU operator time. Width of the band = wall clock.
That is the whole discrepancy between the two metrics, in one picture.

  pixi run -e duckdb-python python scripts/plot_task_concurrency.py
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
WANT_EXEC = 2
RUNS = [("no cache", "WC_cb_base"), ("page cache (CELEBI)", "WC_cb_cache")]
# Operators that take real time; everything else folds into "other".
SHOWN = ["GPU_SCAN", "PROJECTION", "HASH_GROUP_BY", "MERGE_GROUP_BY"]
COLOURS = {"GPU_SCAN": "#b4522e", "PROJECTION": "#e0a33e",
           "HASH_GROUP_BY": "#3d6b66", "MERGE_GROUP_BY": "#7fa6a1",
           "other": "#d8dcda"}
LABEL = {"GPU_SCAN": "scan", "PROJECTION": "projection",
         "HASH_GROUP_BY": "group by", "MERGE_GROUP_BY": "merge group by",
         "other": "other"}


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

    spans = []          # (start, end, operator)
    qspan = collections.defaultdict(lambda: [None, None])
    for tid, series in tasks.items():
        pid = pipeline_of.get(tid)
        if not pid:
            continue
        label = P.resolve_query_label(pid, plan_query, pipeline_plan, query_label)
        parsed = P.parse_label(label) if label else None
        if not parsed or parsed[2] != WANT_EXEC:
            continue
        qnum = parsed[1]
        series.sort(key=lambda e: e[0])
        chain = pipeline_chain.get(pid, {})
        lo, hi = qspan[qnum]
        qspan[qnum] = [series[0][0] if lo is None else min(lo, series[0][0]),
                       series[-1][0] if hi is None else max(hi, series[-1][0])]
        for i, (ts, st) in enumerate(series):
            if i + 1 >= len(series):
                continue
            if isinstance(st, dict) and "Computing" in st and isinstance(st["Computing"], dict):
                kind = chain.get(st["Computing"].get("current_operator_id"), "other")
                spans.append((ts, series[i + 1][0],
                              kind if kind in SHOWN else "other"))
    return spans, qspan


def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.0), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.42))
    for ax, (title, run) in zip(axes, RUNS):
        spans, qspan = gather(run)
        t0 = min(s for s, _, _ in spans)
        # Step function on a fine grid: count per operator at each sample.
        end = max(e for _, e, _ in spans)
        grid = np.linspace(0, (end - t0) / 1e9, 3000)
        counts = {k: np.zeros_like(grid) for k in SHOWN + ["other"]}
        for s, e, kind in spans:
            a, b = (s - t0) / 1e9, (e - t0) / 1e9
            counts[kind] += ((grid >= a) & (grid < b)).astype(float)
        order = ["other"] + SHOWN
        ax.stackplot(grid, [counts[k] for k in order],
                     colors=[COLOURS[k] for k in order],
                     labels=[LABEL[k] for k in order], linewidth=0)

        total = (end - t0) / 1e9
        area = sum(e - s for s, e, _ in spans) / 1e9
        ax.text(0.995, 0.88, f"wall clock {total:.0f}s   ·   summed {area:.0f}s",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                color="#2f3b3a")
        ax.set_title(title, fontsize=9, pad=3, loc="left")
        ax.set_ylabel("tasks computing", fontsize=8.5)
        ax.set_ylim(0, 9)
        ax.set_xlim(0, total)

        # Name the queries that occupy the most wall clock.
        spans_by_q = sorted(((hi - lo) / 1e9, q, lo, hi) for q, (lo, hi) in qspan.items())
        for dur, q, lo, hi in spans_by_q[-3:]:
            mid = ((lo + hi) / 2 - t0) / 1e9
            ax.annotate(f"q{q}", xy=(mid, 7.4), ha="center", fontsize=8,
                        color="#2f3b3a")
            ax.plot([(lo - t0) / 1e9, (hi - t0) / 1e9], [6.9, 6.9],
                    color="#2f3b3a", linewidth=0.7)
        ps.finish(ax, "tasks computing")

    axes[0].legend(loc="upper left", fontsize=7, ncol=5, frameon=False,
                   bbox_to_anchor=(0.0, 1.30), handlelength=1.0, columnspacing=1.2)
    axes[1].set_xlabel("seconds (ClickBench 43 queries, execution 2)", fontsize=8.5)
    ps.save(fig, "task_concurrency")


if __name__ == "__main__":
    main()
