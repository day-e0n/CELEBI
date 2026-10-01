#!/usr/bin/env python3
"""One query's tasks as a slide-sized Gantt chart: baseline against the page cache.

Takes the heaviest query (q29) and draws its tasks as blocks on a shared time axis,
one row per task, segment per operator. Tasks that sit at the same x are running at
the same moment -- that staircase is the overlap the summed metric adds up.

  pixi run -e duckdb-python python scripts/plot_task_gantt_slide.py [query] [n_tasks]
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
WANT_EXEC = 2
RUNS = [("Baseline", "WC_cb_base"), ("CELEBI", "WC_cb_cache")]
COLOURS = {"GPU_SCAN": "#b4522e", "PROJECTION": "#e0a33e",
           "HASH_GROUP_BY": "#4a6fb5", "MERGE_GROUP_BY": "#7fa6a1"}
SHORT = {"GPU_SCAN": "scan", "PROJECTION": "projection",
         "HASH_GROUP_BY": "group by", "MERGE_GROUP_BY": "merge"}


def gather(run: str, query: str):
    telemetry = next((ROOT / run).glob("*/telemetry_data"))
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "P", here / "parse_quent_operator_breakdown.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    query_label, plan_query, pipeline_plan, pipeline_chain = P.load_telemetry(telemetry)

    # The pipeline's instance_name is the task's identity: the operator chain it runs.
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

    rows = []
    for tid, series in tasks.items():
        pid = pipeline_of.get(tid)
        if not pid:
            continue
        label = P.resolve_query_label(pid, plan_query, pipeline_plan, query_label)
        parsed = P.parse_label(label) if label else None
        if not parsed or parsed[2] != WANT_EXEC or f"q{parsed[1]}" != query:
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
            rows.append((blocks[0][0], blocks, pipeline_name.get(pid, "?")))
    rows.sort(key=lambda r: r[0])
    return rows


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "q29"
    n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    data = {}
    for title, run in RUNS:
        rows = gather(run, query)
        # Keep the longest-running tasks; they carry the query.
        rows = sorted(rows, key=lambda r: -sum(e - s for s, e, _ in r[1]))[:n_show]
        rows.sort(key=lambda r: r[0])
        data[title] = rows
    span = max(max(e for _, b, _ in rows for _, e, _ in b) - min(s for s, _, _ in rows)
               for rows in data.values()) / 1e9

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.2), sharex=True,
                             gridspec_kw=dict(hspace=0.3))
    for ax, (title, _) in zip(axes, RUNS):
        rows = data[title]
        t0 = min(s for s, _, _ in rows)
        for y, (_, blocks, _name) in enumerate(rows):
            for start, end, kind in blocks:
                x, w = (start - t0) / 1e9, (end - start) / 1e9
                ax.barh(y, w, left=x, height=0.62, color=COLOURS[kind],
                        edgecolor="white", linewidth=1.1, zorder=3)
                if w > span * 0.07:
                    ax.text(x + w / 2, y, SHORT[kind], ha="center", va="center",
                            fontsize=7, color="white", zorder=4)
        busy = sum(e - s for _, b, _ in rows for s, e, _ in b) / 1e9
        wall = (max(e for _, b, _ in rows for _, e, _ in b) - t0) / 1e9
        ax.set_title(f"{title}  —  {query},  {len(rows)} tasks shown,  "
                     f"wall {wall:.1f}s  vs  summed {busy:.1f}s",
                     fontsize=9, loc="left", pad=4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([name.replace(" -> ", "→").replace("GPU_SCAN", "scan")
                            .replace("HASH_GROUP_BY", "group by")
                            .replace("PROJECTION", "projection")
                            for _, _, name in rows], fontsize=7.5)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_xlim(0, span * 1.02)
        ax.invert_yaxis()
        ax.tick_params(labelsize=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.xaxis.grid(True, color="0.88", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].legend(handles=[Patch(facecolor=COLOURS[k], label=SHORT[k])
                            for k in ("GPU_SCAN", "PROJECTION", "HASH_GROUP_BY")],
                   loc="lower right", fontsize=7.5, ncol=3, frameon=False,
                   bbox_to_anchor=(1.0, 1.12))
    axes[1].set_xlabel("seconds from the query's first task", fontsize=8.5)
    ps.save(fig, "task_gantt_slide")


if __name__ == "__main__":
    main()
