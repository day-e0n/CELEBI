#!/usr/bin/env python3
"""The whole run as one Gantt chart: q1 to q43, every task, in time order.

One row per task, ordered by when it started, so the chart reads top-to-bottom as
the run proceeds. Blocks are the operators the task computed; the dashed line and
label mark where each query ended.

  pixi run -e duckdb-python python scripts/plot_run_gantt.py <run>
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
from matplotlib.patches import Patch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1] / "experiment"
COLOURS = {"GPU_SCAN": "#b4522e", "PROJECTION": "#e0a33e",
           "HASH_GROUP_BY": "#4a6fb5", "MERGE_GROUP_BY": "#7fa6a1",
           "TOP_N": "#8c7fb5", "PARTITION": "#b5a77f",
           "UNGROUPED_AGGREGATE": "#6b8f3d", "MERGE_TOP_N": "#a98fbf",
           "MERGE_AGGREGATE": "#8fae6b", "RESULT_COLLECTOR": "#c9c9c9",
           "FILTER": "#d98f8f", "ORDER_BY": "#8f8fd9", "SORT_SAMPLE": "#8f8fd9",
           "SORT_PARTITION": "#8f8fd9", "MERGE_SORT": "#8f8fd9",
           "STREAMING_LIMIT": "#c9c9c9"}
LEGEND = ["GPU_SCAN", "PROJECTION", "HASH_GROUP_BY", "MERGE_GROUP_BY",
          "TOP_N", "UNGROUPED_AGGREGATE"]
SHORT = {"GPU_SCAN": "scan", "PROJECTION": "projection", "HASH_GROUP_BY": "group by",
         "MERGE_GROUP_BY": "merge group by", "TOP_N": "top n",
         "UNGROUPED_AGGREGATE": "ungrouped agg"}


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "WC_cb_cache"
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

    rows = []
    qend = {}
    for tid, series in tasks.items():
        pid = pipeline_of.get(tid)
        if not pid:
            continue
        label = P.resolve_query_label(pid, plan_query, pipeline_plan, query_label)
        parsed = P.parse_label(label) if label else None
        if not parsed or parsed[2] != 2:
            continue
        qnum = parsed[1]
        series.sort(key=lambda e: e[0])
        blocks = []
        chain = pipeline_chain.get(pid, {})
        for i, (ts, st) in enumerate(series):
            if i + 1 >= len(series):
                continue
            if isinstance(st, dict) and "Computing" in st and isinstance(st["Computing"], dict):
                kind = chain.get(st["Computing"].get("current_operator_id"), "")
                if kind:
                    blocks.append((ts, series[i + 1][0], kind))
        if not blocks:
            continue
        rows.append((blocks[0][0], qnum, blocks))
        last = max(e for _, e, _ in blocks)
        qend[qnum] = max(qend.get(qnum, 0), last)
    rows.sort(key=lambda r: r[0])

    t0 = rows[0][0]
    span = (max(qend.values()) - t0) / 1e9
    height = max(5.0, len(rows) * 0.030)
    fig, ax = plt.subplots(figsize=(14.0, height))

    for y, (_start, qnum, blocks) in enumerate(rows):
        for s, e, kind in blocks:
            x, w = (s - t0) / 1e9, (e - s) / 1e9
            ax.barh(y, max(w, span * 0.0012), left=x, height=0.72,
                    color=COLOURS.get(kind, "#cccccc"), edgecolor="none",
                    linewidth=0, zorder=3)

    # query boundaries
    prev_q = None
    for y, (_s, qnum, _b) in enumerate(rows):
        if prev_q is not None and qnum != prev_q:
            ax.axhline(y - 0.5, color="0.85", linewidth=0.5, zorder=1)
        prev_q = qnum
    # A dashed line at every query end; labels only where there is room for one.
    labelled = sorted(qend.items(), key=lambda kv: kv[1])
    last_x = -1e9
    for qnum, end in labelled:
        x = (end - t0) / 1e9
        room = x - last_x >= span * 0.028
        ax.axvline(x, color="#4a6fb5", linewidth=0.9 if room else 0.5,
                   linestyle="--", zorder=2, alpha=0.8 if room else 0.35)
        if room:
            ax.text(x, -0.018 * len(rows), f"q{qnum}", fontsize=7.5,
                    ha="center", va="bottom", color="#4a6fb5")
            last_x = x

    ax.set_ylim(len(rows) - 0.3, -0.045 * len(rows))
    ax.set_xlim(0, span * 1.005)
    ax.set_ylabel(f"task  (one row each, {len(rows)} tasks, in start order)", fontsize=9)
    ax.set_xlabel("time (seconds into execution 2)", fontsize=9)
    ax.set_yticks([])
    ax.tick_params(labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color="0.93", linewidth=0.5)
    ax.set_axisbelow(True)
    title = "page cache on (CELEBI)" if "cache" in run else "no cache (baseline)"
    ax.set_title(f"{title} — ClickBench q1…q43, execution 2,  wall clock {span:.1f}s",
                 fontsize=10.5, loc="left", pad=22)
    ax.legend(handles=[Patch(facecolor=COLOURS[k], label=SHORT[k]) for k in LEGEND],
              loc="lower left", fontsize=8, ncol=6, frameon=False,
              bbox_to_anchor=(0.0, 1.002), handlelength=1.2)
    ps.save(fig, "run_gantt_cache" if "cache" in run else "run_gantt_base")


if __name__ == "__main__":
    main()
