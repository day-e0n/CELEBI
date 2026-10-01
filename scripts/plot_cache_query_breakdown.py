#!/usr/bin/env python3
"""Where each ClickBench query's time goes with the page cache on.

Left: the operators inside Computing, per query, overlap removed. Two queries carry
the run, so the rest are grouped.
Right: the same queries against the no-cache run, so a bar above zero is a query the
cache made slower.

  pixi run -e duckdb-python python scripts/plot_cache_query_breakdown.py
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
OPS = [("GPU_SCAN", "scan", "#b4522e"), ("PROJECTION", "projection", "#e0a33e"),
       ("HASH_GROUP_BY", "group by", "#4a6fb5"), ("MERGE_GROUP_BY", "merge", "#7fa6a1")]


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

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for tid, series in tasks.items():
        pid = pipeline_of.get(tid)
        if not pid:
            continue
        label = P.resolve_query_label(pid, plan_query, pipeline_plan, query_label)
        parsed = P.parse_label(label) if label else None
        if not parsed or parsed[2] != 2:
            continue
        q = f"q{parsed[1]}"
        series.sort(key=lambda e: e[0])
        chain = pipeline_chain.get(pid, {})
        for i, (ts, st) in enumerate(series):
            if i + 1 >= len(series):
                continue
            span = (ts, series[i + 1][0])
            key = next(iter(st)) if isinstance(st, dict) and st else str(st)
            per[q]["ALL"].append(span)
            if key == "Computing" and isinstance(st["Computing"], dict):
                per[q][chain.get(st["Computing"].get("current_operator_id"), "?")].append(span)
    return per


def main() -> None:
    cache, base = gather("WC_cb_cache"), gather("WC_cb_base")
    order = sorted(cache, key=lambda q: -union_s(cache[q]["ALL"]))
    big, rest = order[:8], order[8:]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.6, 3.4), gridspec_kw=dict(width_ratios=[3, 2], wspace=0.3))

    labels = big + ["other 35"]
    y = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for key, name, colour in OPS:
        vals = [union_s(cache[q].get(key, [])) for q in big]
        vals.append(sum(union_s(cache[q].get(key, [])) for q in rest))
        left.barh(y, vals, 0.62, left=bottom, color=colour, label=name,
                  edgecolor="white", linewidth=0.7, zorder=3)
        for yi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 3:
                left.text(b + v / 2, yi, f"{v:.0f}", ha="center", va="center",
                          fontsize=7, color="white")
        bottom = bottom + np.array(vals)
    for yi, v in enumerate(bottom):
        left.text(v + 0.6, yi, f"{v:.1f}s", va="center", fontsize=7.5, color="#2f3b3a")
    left.set_yticks(y)
    left.set_yticklabels(labels, fontsize=8)
    left.invert_yaxis()
    left.set_xlim(0, max(bottom) * 1.16)
    left.legend(fontsize=7.5, ncol=4, frameon=False, loc="lower right",
                bbox_to_anchor=(1.02, 1.0), handlelength=1.1)
    left.set_xlabel("seconds in Computing (overlap removed)", fontsize=8.5)
    left.set_title("page cache on — where each query's time goes",
                   fontsize=9, loc="left", pad=16)
    ps.finish(left, None)

    deltas = []
    for q in big:
        a, b = union_s(base[q]["ALL"]), union_s(cache[q]["ALL"])
        deltas.append(b - a)
    deltas.append(sum(union_s(cache[q]["ALL"]) - union_s(base[q]["ALL"]) for q in rest))
    colours = ["#b4522e" if d > 0 else "#3d6b66" for d in deltas]
    right.barh(y, deltas, 0.62, color=colours, edgecolor="white", linewidth=0.7, zorder=3)
    for yi, d in enumerate(deltas):
        right.text(d + (0.08 if d > 0 else -0.08), yi, f"{d:+.2f}", va="center",
                   ha="left" if d > 0 else "right", fontsize=7.5)
    right.axvline(0, color="#2f3b3a", linewidth=0.8, zorder=4)
    right.set_yticks(y)
    right.set_yticklabels([])
    right.invert_yaxis()
    lim = max(abs(min(deltas)), max(deltas)) * 1.45
    right.set_xlim(-lim, lim)
    right.set_xlabel("change vs no cache (s)", fontsize=8.5)
    right.set_title("← faster        slower →", fontsize=8.5, loc="center", pad=16)
    ps.finish(right, None)

    ps.save(fig, "cache_query_breakdown")


if __name__ == "__main__":
    main()
