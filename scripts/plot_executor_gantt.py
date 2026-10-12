#!/usr/bin/env python3
"""The four GPU executor threads across a whole run, q1 to q43.

One row per executor thread. A block is a task computing an operator on that
thread; the dashed line is where a query ended and the next one began. Blocks
stacked at the same x on different rows are the overlap.

  pixi run -e duckdb-python python scripts/plot_executor_gantt.py [t_start t_end]
"""
from __future__ import annotations

import collections
import importlib.util
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1] / "experiment"
RUNS = [("no cache", "WC_cb_base"), ("page cache (CELEBI)", "WC_cb_cache")]
if os.environ.get("GANTT_RUNS"):
    a, b = os.environ["GANTT_RUNS"].split(",")
    RUNS = [("no cache", a), ("page cache (CELEBI)", b)]
STEM = os.environ.get("GANTT_STEM", "pipeline_gantt")
OPS = [("GPU_SCAN", "scan", "#b4522e"), ("PROJECTION", "projection", "#e0a33e"),
       ("HASH_GROUP_BY", "group by", "#4a6fb5"), ("MERGE_GROUP_BY", "merge", "#7fa6a1"),
       ("TOP_N", "top n", "#8c7fb5"), ("UNGROUPED_AGGREGATE", "ungrouped agg", "#6b8f3d")]
COLOUR = {k: c for k, _, c in OPS}


def gather(run: str):
    telemetry = next((ROOT / run).glob("*/telemetry_data"))
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "P", here / "parse_quent_operator_breakdown.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    query_label, plan_query, pipeline_plan, pipeline_chain = P.load_telemetry(telemetry)

    tasks = collections.defaultdict(list)
    pipeline_name = {}
    for path in sorted(telemetry.rglob("operator/*.ndjson")):
        with path.open() as handle:
            for line in handle:
                obj = json.loads(line)
                decl = obj.get("data", {}).get("Declaration")
                if decl:
                    pipeline_name[obj["id"]] = decl.get("instance_name", "")

    pipeline_of = {}
    for path in sorted(telemetry.rglob("task/*.ndjson")):
        with path.open() as handle:
            for line in handle:
                obj = json.loads(line)
                st = obj.get("data", {}).get("state")
                tasks[obj["id"]].append((obj["timestamp"], st))
                if isinstance(st, dict) and "Created" in st and isinstance(st["Created"], dict):
                    pipeline_of[obj["id"]] = st["Created"].get("pipeline_uuid", "")

    blocks = []        # (thread_id, start, end, operator, query)
    qend = {}
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
        for i, (ts, st) in enumerate(series):
            if i + 1 >= len(series):
                continue
            if isinstance(st, dict) and "Computing" in st and isinstance(st["Computing"], dict):
                c = st["Computing"]
                kind = chain.get(c.get("current_operator_id"), "")
                if kind in COLOUR:
                    blocks.append((pipeline_name.get(pid, "?"),
                                   ts, series[i + 1][0], kind, q))
        last = max(e for _, e in [(0, s[0]) for s in [series[-1]]])
        qend[q] = max(qend.get(q, 0), series[-1][0])
    return blocks, qend


def main() -> None:
    window = None
    if len(sys.argv) > 2:
        window = (float(sys.argv[1]), float(sys.argv[2]))

    width = 16.0 if not window else max(16.0, (window[1] - window[0]) * 1.15)
    fig, axes = plt.subplots(2, 1, figsize=(width, 8.4), sharex=True,
                             gridspec_kw=dict(hspace=0.55))
    for ax, (title, run) in zip(axes, RUNS):
        blocks, qend = gather(run)
        t0 = min(s for _, s, _, _, _ in blocks)
        # Group tasks by the operator chain they run; busiest chain on top.
        busy = collections.Counter()
        for name, s_, e_, _k, _q in blocks:
            busy[name] += e_ - s_
        threads = [n for n, _ in busy.most_common()]
        row = {t: i for i, t in enumerate(threads)}
        span = (max(e for _, _, e, _, _ in blocks) - t0) / 1e9

        for thread, s, e, kind, _q in blocks:
            x, w = (s - t0) / 1e9, (e - s) / 1e9
            ax.barh(row[thread], max(w, span * 0.0009), left=x, height=0.68,
                    color=COLOUR[kind], edgecolor="none", zorder=3)

        lo, hi = window if window else (0, span)
        tier = 0
        last = [-1e9, -1e9]
        for q, end in sorted(qend.items(), key=lambda kv: kv[1]):
            x = (end - t0) / 1e9
            if not (lo <= x <= hi):
                continue
            ax.axvline(x, color="#4a6fb5", linestyle="--", linewidth=0.8,
                       alpha=0.7, zorder=2)
            # Alternate between two label rows so neighbouring ends never collide.
            tier = 0 if x - last[0] >= (hi - lo) * 0.016 else 1
            ax.text(x, len(threads) - 0.25 + tier * 0.45, f"q{q}", fontsize=7.5,
                    ha="center", va="bottom", color="#4a6fb5")
            last[tier] = x

        def pretty(name):
            n = re.sub(r"\((\d+)\)", "", name)
            n = (n.replace("GPU_SCAN", "scan").replace("HASH_GROUP_BY", "group by")
                 .replace("PROJECTION", "proj").replace("MERGE_GROUP_BY", "merge gb")
                 .replace("UNGROUPED_AGGREGATE", "ungrouped agg")
                 .replace("MERGE_AGGREGATE", "merge agg")
                 .replace("RESULT_COLLECTOR", "result").replace("MERGE_TOP_N", "merge topn")
                 .replace("TOP_N", "top n").replace("PARTITION", "partition")
                 .replace(" -> ", "→"))
            return n.strip()
        ax.set_yticks(range(len(threads)))
        ax.set_yticklabels([pretty(t) for t in threads], fontsize=8.5)
        ax.set_ylim(-0.6, len(threads) + 0.5)
        ax.set_xlim(lo, hi)
        ax.invert_yaxis()
        ax.set_ylabel("task", fontsize=9)
        ax.set_title(f"{title}   —   wall clock {span:.1f}s", fontsize=10, loc="left", pad=14)
        ax.tick_params(labelsize=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.xaxis.grid(True, color="0.93", linewidth=0.5)
        ax.set_axisbelow(True)

    axes[1].set_xlabel("time (seconds into execution 2)", fontsize=9)
    fig.legend(handles=[Patch(facecolor=c, label=n) for _, n, c in OPS],
               loc="upper center", ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    stem = STEM if not window else f"{STEM}_{int(window[0])}_{int(window[1])}"
    ps.save(fig, stem)


if __name__ == "__main__":
    main()
