#!/usr/bin/env python3
"""The cache-on run as a Gantt chart: the queries that matter, on one time axis.

Each row is a task, labelled with the query it belongs to and the operator chain it
runs. Blocks are the operators it computed; gaps between blocks are the task waiting
(Queued or Reserving). Rows that line up vertically were running at the same moment.

  pixi run -e duckdb-python python scripts/plot_cache_gantt.py [run]
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
           "TOP_N": "#8c7fb5", "PARTITION": "#b5a77f"}
SHORT = {"GPU_SCAN": "scan", "PROJECTION": "proj", "HASH_GROUP_BY": "group by",
         "MERGE_GROUP_BY": "merge", "TOP_N": "top n", "PARTITION": "part"}
# Queries worth drawing, and how many of their tasks to show.
PICK = [("q29", 4), ("q28", 4), ("q1", 3), ("q24", 3), ("q23", 2)]


def gather(run: str):
    telemetry = next((ROOT / run).glob("*/telemetry_data"))
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "P", here / "parse_quent_operator_breakdown.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    query_label, plan_query, pipeline_plan, pipeline_chain = P.load_telemetry(telemetry)
    pipeline_name = {}
    for path in sorted(telemetry.rglob("operator/*.ndjson")):
        with path.open() as handle:
            for line in handle:
                obj = json.loads(line)
                decl = obj.get("data", {}).get("Declaration")
                if decl:
                    pipeline_name[obj["id"]] = decl.get("instance_name", "")

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

    rows = collections.defaultdict(list)
    for tid, series in tasks.items():
        pid = pipeline_of.get(tid)
        if not pid:
            continue
        label = P.resolve_query_label(pid, plan_query, pipeline_plan, query_label)
        parsed = P.parse_label(label) if label else None
        if not parsed or parsed[2] != 2:
            continue
        series.sort(key=lambda e: e[0])
        chain = pipeline_chain.get(pid, {})
        blocks = []
        for i, (ts, st) in enumerate(series):
            if i + 1 >= len(series):
                continue
            if isinstance(st, dict) and "Computing" in st and isinstance(st["Computing"], dict):
                kind = chain.get(st["Computing"].get("current_operator_id"))
                if kind in COLOURS:
                    blocks.append((ts, series[i + 1][0], kind))
        if blocks:
            rows[f"q{parsed[1]}"].append((blocks[0][0], blocks, pipeline_name.get(pid, "")))
    return rows


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "WC_cb_cache"
    rows = gather(run)

    picked = []
    for q, n in PICK:
        tasks = sorted(rows.get(q, []), key=lambda r: -sum(e - s for s, e, _ in r[1]))[:n]
        for start, blocks, name in sorted(tasks, key=lambda r: r[0]):
            picked.append((q, start, blocks, name))
    if not picked:
        raise SystemExit("no tasks")
    t0 = min(s for _, s, _, _ in picked)
    t_end = max(e for _, _, b, _ in picked for _, e, _ in b)
    span = (t_end - t0) / 1e9

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    labels = []
    for y, (q, _start, blocks, name) in enumerate(picked):
        for start, end, kind in blocks:
            x, w = (start - t0) / 1e9, (end - start) / 1e9
            ax.barh(y, w, left=x, height=0.66, color=COLOURS[kind],
                    edgecolor="white", linewidth=1.0, zorder=3)
            if w > span * 0.035:
                ax.text(x + w / 2, y, SHORT[kind], ha="center", va="center",
                        fontsize=7, color="white", zorder=4)
        chain = (name.replace("GPU_SCAN", "scan").replace("HASH_GROUP_BY", "group by")
                 .replace("PROJECTION", "proj").replace("MERGE_GROUP_BY", "merge")
                 .replace(" -> ", "→"))
        import re
        chain = re.sub(r"\(\d+\)", "", chain)
        labels.append(f"{q}   {chain}")

    # Separators between queries
    seen = []
    for y, (q, _s, _b, _n) in enumerate(picked):
        if not seen or seen[-1] != q:
            if seen:
                ax.axhline(y - 0.5, color="0.8", linewidth=0.7, zorder=1)
            seen.append(q)

    ax.set_yticks(range(len(picked)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylim(-0.7, len(picked) - 0.3)
    ax.set_xlim(0, span * 1.01)
    ax.invert_yaxis()
    ax.tick_params(labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("seconds into execution 2  (gaps = task waiting for memory or queue)",
                  fontsize=8.5)
    title = "page cache on (CELEBI)" if "cache" in run else "no cache"
    ax.set_title(f"{title} — ClickBench, the five queries that carry the run",
                 fontsize=9.5, loc="left", pad=18)
    ax.legend(handles=[Patch(facecolor=COLOURS[k], label=SHORT[k])
                       for k in ("GPU_SCAN", "PROJECTION", "HASH_GROUP_BY", "MERGE_GROUP_BY")],
              loc="lower left", fontsize=7.5, ncol=4, frameon=False,
              bbox_to_anchor=(0.0, 1.0), handlelength=1.1)
    ps.save(fig, "cache_gantt" if "cache" in run else "base_gantt")


if __name__ == "__main__":
    main()
