#!/usr/bin/env python3
"""q1 to q43 in execution order, one row per query, cache off vs cache on.

Each row collapses that query's tasks into one band: the operator segments are the
union of what its tasks were computing, so a row's width is the query's own elapsed
time. Rows are stacked in the order the benchmark ran them.

The cache-on panel marks each query by whether the page cache served it anything.

  pixi run -e duckdb-python python scripts/plot_query_gantt.py
"""
from __future__ import annotations

import collections
import importlib.util
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1] / "experiment"
OPS = [("GPU_SCAN", "scan", "#b4522e"),
       ("PROJECTION", "projection", "#e0a33e"),
       ("HASH_GROUP_BY", "group by", "#4a6fb5"),
       ("MERGE_GROUP_BY", "merge", "#7fa6a1"),
       ("TOP_N", "top n", "#8c7fb5"),
       ("UNGROUPED_AGGREGATE", "ungrouped agg", "#6b8f3d")]
COLOUR = {k: c for k, _, c in OPS}
SERVE = re.compile(r"\[page-trace\] serve col='([^']+)'")
LABELMARK = re.compile(r"==== query (\S+) ====")


def union(items):
    merged = []
    for s, e in sorted(items):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


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

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    bounds = {}
    for tid, series in tasks.items():
        pid = pipeline_of.get(tid)
        if not pid:
            continue
        label = P.resolve_query_label(pid, plan_query, pipeline_plan, query_label)
        parsed = P.parse_label(label) if label else None
        if not parsed or parsed[2] != 2:
            continue
        q = parsed[1]
        series.sort(key=lambda e: e[0])
        chain = pipeline_chain.get(pid, {})
        lo, hi = bounds.get(q, (series[0][0], series[-1][0]))
        bounds[q] = (min(lo, series[0][0]), max(hi, series[-1][0]))
        for i, (ts, st) in enumerate(series):
            if i + 1 >= len(series):
                continue
            if isinstance(st, dict) and "Computing" in st and isinstance(st["Computing"], dict):
                kind = chain.get(st["Computing"].get("current_operator_id"), "")
                if kind in COLOUR:
                    per[q][kind].append((ts, series[i + 1][0]))
    return per, bounds


def elapsed(bounds):
    return {q: (hi - lo) / 1e9 for q, (lo, hi) in bounds.items()}


def main() -> None:
    runs = [("no cache", "WC_cb_base"), ("page cache (CELEBI)", "WC_cb_cache")]
    data = {lab: gather(run) for lab, run in runs}
    order = sorted(data[runs[0][0]][1], key=lambda q: data[runs[0][0]][1][q][0])
    span = max(max(hi for _, hi in b.values()) - min(lo for lo, _ in b.values())
               for _, b in data.values()) / 1e9

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 7.0), sharey=True,
                             gridspec_kw=dict(wspace=0.06))
    for ax, (lab, _run) in zip(axes, runs):
        per, bounds = data[lab]
        t0 = min(lo for lo, _ in bounds.values())
        for y, q in enumerate(order):
            for key, _n, colour in OPS:
                for s, e in union(per[q].get(key, [])):
                    x, w = (s - t0) / 1e9, (e - s) / 1e9
                    ax.barh(y, max(w, span * 0.0015), left=x, height=0.7,
                            color=colour, edgecolor="none", zorder=3)
            lo, hi = bounds[q]
            ax.text((hi - t0) / 1e9 + span * 0.008, y, f"{(hi-lo)/1e9:.1f}s",
                    va="center", fontsize=6, color="#6b7572")
        ax.set_xlim(0, span * 1.08)
        ax.set_ylim(len(order) - 0.4, -0.6)
        ax.set_xlabel("time (s)", fontsize=9)
        ax.set_title(lab, fontsize=10, loc="left", pad=4)
        ax.tick_params(labelsize=7.5)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.xaxis.grid(True, color="0.92", linewidth=0.5)
        ax.set_axisbelow(True)

    # Mark each query by what the cache did to its elapsed time.
    e0 = elapsed(data[runs[0][0]][1])
    e1 = elapsed(data[runs[1][0]][1])
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"q{q}" for q in order], fontsize=7)
    for y, q in enumerate(order):
        d = e1.get(q, 0) - e0.get(q, 0)
        if d < -0.05:
            axes[1].text(span * 1.005, y, f"▼ {d:.2f}s", fontsize=6.5, va="center",
                         color="#3d6b66")
        elif d > 0.05:
            axes[1].text(span * 1.005, y, f"▲ +{d:.2f}s", fontsize=6.5, va="center",
                         color="#b4522e")
    axes[1].text(span * 1.005, -1.6, "▼ faster   ▲ slower", fontsize=7,
                 color="#2f3b3a", va="center")
    axes[0].set_ylabel("query, in execution order", fontsize=9)
    fig.legend(handles=[Patch(facecolor=c, label=n) for _, n, c in OPS],
               loc="upper center", ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    ps.save(fig, "query_gantt")


if __name__ == "__main__":
    main()
