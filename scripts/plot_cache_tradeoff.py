#!/usr/bin/env python3
"""What the page cache actually trades: GPU work for memory, at no gain in latency.

The left panel is the metric these experiments have used throughout -- GPU operator
time summed over tasks. By it the cache is a large win, and the win is real in the
sense it measures: each task's Computing interval shrinks from 269.9ms to 153.7ms.

The right panel is why that never reached the user. Those intervals overlap, and the
cache holds 8.79GB of device memory, so fewer tasks get a reservation and run at once
(2.85 -> 1.63). Occupancy and wall clock both come out flat.

  pixi run -e duckdb-python python scripts/plot_cache_tradeoff.py
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1] / "experiment"


def task_spans(run: str) -> list[tuple[int, int]]:
    """Every interval a task spent in Computing, across all executions."""
    events: dict[str, list] = collections.defaultdict(list)
    telemetry = next((ROOT / run).glob("*/telemetry_data"))
    for path in sorted(telemetry.rglob("task/*.ndjson")):
        with path.open() as handle:
            for line in handle:
                obj = json.loads(line)
                events[obj["id"]].append((obj["timestamp"], obj.get("data", {}).get("state")))
    spans = []
    for series in events.values():
        series.sort(key=lambda e: e[0])
        for i, (ts, state) in enumerate(series):
            if isinstance(state, dict) and "Computing" in state and i + 1 < len(series):
                spans.append((ts, series[i + 1][0]))
    return spans


def union_seconds(spans: list[tuple[int, int]]) -> float:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(e - s for s, e in merged) / 1e9


def summed_seconds(run: str) -> float:
    """The headline metric: bucket.csv total_ms over warm executions."""
    per: dict[int, float] = collections.defaultdict(float)
    with (ROOT / run / "bucket.csv").open() as handle:
        for row in csv.DictReader(handle):
            if int(row["execution"]) > 1:
                per[int(row["execution"])] += float(row["total_ms"]) / 1000.0
    return sum(per.values()) / len(per)


ORDERS = [("natural", "F_cb_cache", "F_cb_reorder"),
          ("worst_case", "S_cb_worst_c", "S_cb_worst_r"),
          ("random_a", "T_cb_randa_c", "T_cb_randa_r"),
          ("random_b", "T_cb_randb_c", "T_cb_randb_r"),
          ("random_c", "T_cb_randc_c", "T_cb_randc_r")]


def main() -> None:
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[5, 3], wspace=0.32))

    # --- left: the summed metric, five arrival orders --------------------
    x = np.arange(len(ORDERS))
    width = 0.36
    plain = [summed_seconds(c) for _, c, _ in ORDERS]
    reord = [summed_seconds(r) for _, _, r in ORDERS]
    left.bar(x - width / 2, plain, width, label="cache on", zorder=3, **ps.LIGHT)
    left.bar(x + width / 2, reord, width, label="cache on + reorder", zorder=3, **ps.DARK)
    base = summed_seconds("F_cb_base")
    left.axhline(base, color=ps.ACCENT["facecolor"], linewidth=0.9, linestyle="--", zorder=4)
    left.text(-0.42, base - 6, f"no cache ({base:.0f}s)", ha="left", va="top",
              fontsize=7.5, color=ps.ACCENT["facecolor"])
    left.set_xticks(x)
    left.set_xticklabels([n for n, _, _ in ORDERS], fontsize=7.5)
    left.set_ylim(0, base * 1.22)
    left.legend(loc="upper left", fontsize=7.5, ncol=2, handlelength=1.2,
                bbox_to_anchor=(0.0, 1.0))
    left.set_xlabel("ClickBench 43 queries — arrival order", fontsize=8.5, labelpad=3)
    ps.finish(left, "GPU operator time, summed (s)")

    # --- right: the same two runs on four measures -----------------------
    base_spans, cache_spans = task_spans("WC_cb_base"), task_spans("WC_cb_cache")
    metrics = [
        ("per-task\nComputing",
         sum(e - s for s, e in base_spans) / len(base_spans) / 1e6,
         sum(e - s for s, e in cache_spans) / len(cache_spans) / 1e6, "ms"),
        ("summed\nGPU time", sum(e - s for s, e in base_spans) / 1e9,
         sum(e - s for s, e in cache_spans) / 1e9, "s"),
        ("tasks running\nat once",
         (sum(e - s for s, e in base_spans) / 1e9) / union_seconds(base_spans),
         (sum(e - s for s, e in cache_spans) / 1e9) / union_seconds(cache_spans), ""),
        ("wall clock", 76.56, 76.14, "s"),
    ]
    xs = np.arange(len(metrics))
    # Normalise each measure to its baseline so four different units share one axis.
    rel = [b / a for _, a, b, _ in metrics]
    right.axhline(1.0, color="0.55", linewidth=0.7, zorder=2)
    right.bar(xs, rel, 0.5, zorder=3,
              **(ps.DARK if True else ps.LIGHT))
    for xi, (name, a, b, unit) in zip(xs, metrics):
        right.text(xi, rel[xi] + 0.03, f"{(rel[xi]-1)*100:+.0f}%", ha="center",
                   va="bottom", fontsize=7.5,
                   color="#b4522e" if rel[xi] > 1.02 else "#2f3b3a")
        right.text(xi, 0.06, f"{a:.0f}{unit}→{b:.0f}{unit}" if unit != ""
                   else f"{a:.2f}→{b:.2f}", ha="center", va="bottom",
                   fontsize=6.5, color="white", rotation=90)
    right.set_xticks(xs)
    right.set_xticklabels([m[0] for m in metrics], fontsize=7)
    right.set_ylim(0, 1.35)
    right.set_ylabel("relative to no cache", fontsize=8.5)
    right.set_xlabel("cache on vs no cache, natural order", fontsize=8, labelpad=3)
    ps.finish(right, None)

    ps.save(fig, "cache_tradeoff")


if __name__ == "__main__":
    main()
