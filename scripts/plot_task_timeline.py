#!/usr/bin/env python3
"""Every task of a ClickBench run on one time axis, no overlap removed.

One row per task, sorted by when it was created; colour is the state it was in.
Overlap is the point: wherever rows stack vertically at the same x, that many
tasks were in flight at once. The summed operator-time metric counts all of that
vertical extent; the wall clock is the horizontal extent.

  pixi run -e duckdb-python python scripts/plot_task_timeline.py
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
COLOURS = {"Queued": "#d8dcda", "Reserving": "#b4522e", "Computing": "#3d6b66",
           "Preparing": "#7fa6a1", "Downgrading": "#e0a33e"}
RUNS = [("no cache", "WC_cb_base"), ("page cache (CELEBI)", "WC_cb_cache")]


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

    rows = []
    for tid, series in tasks.items():
        pid = pipeline_of.get(tid)
        if not pid:
            continue
        label = P.resolve_query_label(pid, plan_query, pipeline_plan, query_label)
        parsed = P.parse_label(label) if label else None
        if not parsed or parsed[2] != WANT_EXEC:
            continue
        series.sort(key=lambda e: e[0])
        spans = []
        for i, (ts, st) in enumerate(series):
            if i + 1 >= len(series):
                continue
            key = next(iter(st)) if isinstance(st, dict) and st else str(st)
            spans.append((ts, series[i + 1][0], key))
        if spans:
            rows.append((series[0][0], spans, parsed[1]))
    rows.sort(key=lambda r: r[0])
    return rows


def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True,
                             gridspec_kw=dict(hspace=0.18))
    for ax, (label, run) in zip(axes, RUNS):
        rows = gather(run)
        t0 = rows[0][0]
        for y, (_, spans, _qnum) in enumerate(rows):
            for start, end, state in spans:
                width = (end - start) / 1e9
                if width <= 0:
                    continue
                ax.barh(y, width, left=(start - t0) / 1e9, height=1.0,
                        color=COLOURS.get(state, "#f0f0f0"), linewidth=0)
        ax.set_ylim(-2, len(rows) + 2)
        ax.set_ylabel(f"{label}\n{len(rows)} tasks", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="0.9", linewidth=0.5)
        total = (max(e for _, s, _ in rows for _, e, _ in s) - t0) / 1e9
        ax.text(0.995, 0.04, f"wall clock {total:.1f}s", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8)

    axes[1].set_xlabel("seconds from first task (ClickBench 43 queries, execution 2)",
                       fontsize=8.5)
    axes[0].legend(handles=[Patch(facecolor=COLOURS[k], label=k, edgecolor="none")
                            for k in ("Queued", "Reserving", "Computing")],
                   loc="upper left", fontsize=7.5, ncol=3, frameon=False,
                   bbox_to_anchor=(0.0, 1.14))
    ps.save(fig, "task_timeline")


if __name__ == "__main__":
    main()
