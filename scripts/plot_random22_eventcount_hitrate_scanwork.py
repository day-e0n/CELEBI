#!/usr/bin/env python3
"""Combined 2-panel figure for the clean random22_breakdown_sf50_10x dataset:
PER-QUERY (not cumulative) EVENT-COUNT eligible cache hit rate (left) and
per-query scan time (right), by query position (fixed q1..q22 order), for 3
stages (Baseline / Baseline+page caching / CELEBI). Each point is that single
query's own value (mean over hot executions for scan time) -- not a running
sum/average, so unlike a cumulative-ratio plot there's no ambiguity about why
a line moves up or down between points."""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from plot_cache_hit_rate_eligible import stage_event_series  # noqa: E402
from cache_aware_query_reorder import ReorderConfig, reorder_query_sequence  # noqa: E402

ARRIVAL_ORDER = [8, 1, 16, 3, 14, 4, 12, 19, 20, 15, 9, 17, 6, 11, 2, 22, 7, 13, 10, 18, 5, 21]
DATA_ROOT = REPO_ROOT / "experiment" / "random22_breakdown_sf50_10x"
EXECUTIONS = 10

_cfg = ReorderConfig(policy="fixed-overlap", scope="fixed_width", window=0,
                      keep_first=False, resident_column_budget=0)
CELEBI_ORDER = list(reorder_query_sequence(ARRIVAL_ORDER, _cfg).reordered_queries)

STAGES = [
    ("Baseline", "cold_hot", ARRIVAL_ORDER),
    ("Baseline+page caching", "basepaging", ARRIVAL_ORDER),
    ("Baseline+page caching+reordering (CELEBI)", "celebi", CELEBI_ORDER),
]


def mean_scan_ms_by_qnum(bucket_csv: Path) -> dict[str, float]:
    """query -> mean scan ms across hot executions (execution != 1), matching
    build_operator_breakdown_comparison.py's steady_state_series rule."""
    grouped: dict[str, list[float]] = defaultdict(list)
    with bucket_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["execution"]) == 1:
                continue
            grouped[f"q{row['query']}" if row["query"].isdigit() else row["query"]].append(float(row["scan"]))
    return {q: statistics.mean(vals) for q, vals in grouped.items()}


def per_query_values(
    hit_series: list[tuple[str, int, int, int]], scan_by_q: dict[str, float], x_labels: list[str]
) -> tuple[list[float], list[float]]:
    """Each query's OWN hit rate / scan time -- no accumulation across queries."""
    hit_by_q = {q: (b, m, x) for q, b, m, x in hit_series}
    hit_rate, scan_work = [], []
    for q in x_labels:
        b, m, _x = hit_by_q.get(q, (0, 0, 0))
        eligible = b + m  # excludes dynamic_filter_scan (x), same convention as stage_eligible_series
        hit_rate.append(b / eligible * 100.0 if eligible else 0.0)
        scan_work.append(scan_by_q.get(q, 0.0) / 1000.0)
    return hit_rate, scan_work


def deconflicted_label_ys(finals: list[float], min_frac: float = 0.06) -> list[float]:
    """End-of-line labels collide when two series finish within a few percent
    of each other (e.g. Baseline vs Baseline+page caching). Push the lower one
    down by a minimum gap (as a fraction of the largest final value) so labels
    stay legible without moving the actual data points/markers."""
    span = max(finals) if finals else 1.0
    min_gap = span * min_frac
    order = sorted(range(len(finals)), key=lambda i: -finals[i])
    label_ys = list(finals)
    for k in range(1, len(order)):
        prev, cur = order[k - 1], order[k]
        if label_ys[prev] - label_ys[cur] < min_gap:
            label_ys[cur] = label_ys[prev] - min_gap
    return label_ys


def cumulative_fixed_denom_values(
    hit_series: list[tuple[str, int, int, int]], scan_by_q: dict[str, float], x_labels: list[str]
) -> tuple[list[float], list[float]]:
    """Monotonic cumulative hit rate: numerator accumulates as usual, but the
    denominator is FIXED at the sequence's total eligible count (not
    recomputed at each step), so the line only rises -- it reads as "how much
    of the eventual total hit-rate has been accumulated by query k",
    unlike a running average of ratios (which can dip when a later item's own
    rate is below the average so far)."""
    hit_by_q = {q: (b, m, x) for q, b, m, x in hit_series}
    final_eligible = sum(b + m for b, m, _x in hit_by_q.values())
    hit_rate, scan_work = [], []
    cbacked = cscan = 0.0
    for q in x_labels:
        b, m, _x = hit_by_q.get(q, (0, 0, 0))
        cbacked += b
        hit_rate.append(cbacked / final_eligible * 100.0 if final_eligible else 0.0)
        cscan += scan_by_q.get(q, 0.0) / 1000.0
        scan_work.append(cscan)
    return hit_rate, scan_work


def main() -> int:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    colors = ["#4c78a8", "#59a14f", "#f2b134", "#e45756"]
    ink_primary = "#0b0b0b"

    x_labels = [f"q{i}" for i in range(1, 23)]

    stage_data = []
    for label, tag, query_order in STAGES:
        log_dir = DATA_ROOT / tag / "log_dir"
        hit_series = stage_event_series(log_dir, [f"q{q}" for q in query_order], expected_iterations=EXECUTIONS)
        scan_by_q = mean_scan_ms_by_qnum(DATA_ROOT / f"{tag}_bucket.csv")
        hit_rate, scan_work = per_query_values(hit_series, scan_by_q, x_labels)
        stage_data.append((label, hit_rate, scan_work))
        avg_rate = statistics.mean(hit_rate)
        total_scan = sum(scan_work)
        print(f"{label}: mean per-query eligible hit rate = {avg_rate:.2f}%, "
              f"total scan time (sum over 22 queries) = {total_scan:.1f}s")

    n = len(x_labels)
    x = np.arange(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5))

    for idx, (label, hit_rate, _) in enumerate(stage_data):
        ax1.plot(x, hit_rate, color=colors[idx % len(colors)], linewidth=2.0,
                  marker="o", markersize=6, label=label)
    ax1.set_xlabel("TPC-H query", fontsize=18, color=ink_primary)
    ax1.set_ylabel("Per-query cache hit ratio\namong cacheable scans (%)", fontsize=18, color=ink_primary)
    ax1.tick_params(axis="y", colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    ax1.tick_params(axis="x", colors=ink_primary, labelcolor=ink_primary)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=10, color=ink_primary, rotation=90)
    ymax = max(v for _, hit_rate, _ in stage_data for v in hit_rate)
    ax1.set_ylim(0, ymax * 1.15)
    ax1.set_xlim(0, n + 1)
    ax1.grid(axis="y", alpha=0.2)

    for idx, (label, _, scan_work) in enumerate(stage_data):
        ax2.plot(x, scan_work, color=colors[idx % len(colors)], linewidth=2.0,
                  marker="o", markersize=6, label=label)
    ax2.set_xlabel("TPC-H query", fontsize=18, color=ink_primary)
    ax2.set_ylabel("Per-query scan time (s)\n(mean over hot executions)", fontsize=18, color=ink_primary)
    ax2.tick_params(axis="y", colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    ax2.tick_params(axis="x", colors=ink_primary, labelcolor=ink_primary)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=10, color=ink_primary, rotation=90)
    ax2.set_xlim(0, n + 1)
    ax2.grid(axis="y", alpha=0.2)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=True, edgecolor=ink_primary, fontsize=17,
               loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=len(stage_data),
               labelcolor=ink_primary)

    fig.tight_layout()
    out_png = REPO_ROOT / "experiment" / "graph" / "random22_sf50_3stage_perquery_eventcount_hitrate_scanwork.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {out_png}")

    # --- second figure: monotonic (fixed-denominator) cumulative version ---
    cum_data = []
    for label, tag, query_order in STAGES:
        log_dir = DATA_ROOT / tag / "log_dir"
        hit_series = stage_event_series(log_dir, [f"q{q}" for q in query_order], expected_iterations=EXECUTIONS)
        scan_by_q = mean_scan_ms_by_qnum(DATA_ROOT / f"{tag}_bucket.csv")
        hit_rate, scan_work = cumulative_fixed_denom_values(hit_series, scan_by_q, x_labels)
        cum_data.append((label, hit_rate, scan_work))

    fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(15.5, 6.5))

    for idx, (label, hit_rate, _) in enumerate(cum_data):
        bx1.plot(x, hit_rate, color=colors[idx % len(colors)], linewidth=2.4,
                  marker="o", markersize=5, label=label)
        bx1.annotate(f"{hit_rate[-1]:.1f}%", (x[-1], hit_rate[-1]), textcoords="offset points",
                     xytext=(8, 0), ha="left", va="center", fontsize=12, color=ink_primary)
    bx1.set_xlabel("TPC-H query", fontsize=18, color=ink_primary)
    bx1.set_ylabel("Cumulative cache hit ratio\namong cacheable scans (%)", fontsize=18, color=ink_primary)
    bx1.tick_params(axis="y", colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    bx1.tick_params(axis="x", colors=ink_primary, labelcolor=ink_primary)
    bx1.set_xticks(x)
    bx1.set_xticklabels(x_labels, fontsize=10, color=ink_primary, rotation=90)
    ymax2 = max(v for _, hit_rate, _ in cum_data for v in hit_rate)
    bx1.set_ylim(0, ymax2 * 1.3)
    bx1.set_xlim(0, n + 3)
    bx1.grid(axis="y", alpha=0.2)

    baseline_final2 = cum_data[0][2][-1]
    finals2 = [scan_work[-1] for _, _, scan_work in cum_data]
    label_ys2 = deconflicted_label_ys(finals2)
    label_ys2[0] += 3.5  # lift Baseline label a bit further above its line
    label_ys2[1] += 3.5  # lift Baseline+page caching label to match
    for idx, (label, _, scan_work) in enumerate(cum_data):
        bx2.plot(x, scan_work, color=colors[idx % len(colors)], linewidth=2.4,
                  marker="o", markersize=5, label=label)
        final = scan_work[-1]
        pct = (baseline_final2 - final) / baseline_final2 * 100.0 if baseline_final2 else 0.0
        text = f"{final:.1f}s" if idx == 0 else f"{final:.1f}s ({pct:+.1f}%)"
        bx2.annotate(text, xy=(x[-1], final), xytext=(x[-1] + 0.5, label_ys2[idx]),
                     textcoords="data", ha="left", va="center", fontsize=12, color=ink_primary)
    bx2.set_ylim(top=max(finals2) * 1.14)
    bx2.set_xlabel("TPC-H query", fontsize=18, color=ink_primary)
    bx2.set_ylabel("Cumulative scan work (s)", fontsize=18, color=ink_primary)
    bx2.tick_params(axis="y", colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    bx2.tick_params(axis="x", colors=ink_primary, labelcolor=ink_primary)
    bx2.set_xticks(x)
    bx2.set_xticklabels(x_labels, fontsize=10, color=ink_primary, rotation=90)
    bx2.set_xlim(0, n + 3)
    bx2.grid(axis="y", alpha=0.2)

    handles2, labels2 = bx1.get_legend_handles_labels()
    fig2.legend(handles2, labels2, frameon=True, edgecolor=ink_primary, fontsize=17,
                loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=len(cum_data),
                labelcolor=ink_primary)

    fig2.tight_layout()
    out_png2 = REPO_ROOT / "experiment" / "graph" / "random22_sf50_3stage_cumulative_fixeddenom_eventcount_hitrate_scanwork.png"
    fig2.savefig(out_png2, dpi=200, bbox_inches="tight")
    print(f"wrote {out_png2}")

    # --- third figure: per-query (non-cumulative) hit rate + cumulative scan work ---
    fig3, (cx1, cx2) = plt.subplots(1, 2, figsize=(15.5, 6.5))

    for idx, (label, hit_rate, _) in enumerate(stage_data):  # stage_data: per-query hit_rate
        cx1.plot(x, hit_rate, color=colors[idx % len(colors)], linewidth=2.0,
                  marker="o", markersize=6, label=label)
    cx1.set_xlabel("TPC-H query", fontsize=18, color=ink_primary)
    cx1.set_ylabel("Per-query cache hit ratio\namong cacheable scans (%)", fontsize=18, color=ink_primary)
    cx1.tick_params(axis="y", colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    cx1.tick_params(axis="x", colors=ink_primary, labelcolor=ink_primary)
    cx1.set_xticks(x)
    cx1.set_xticklabels(x_labels, fontsize=10, color=ink_primary, rotation=90)
    cx1.set_ylim(0, max(v for _, hit_rate, _ in stage_data for v in hit_rate) * 1.15)
    cx1.set_xlim(0, n + 1)
    cx1.grid(axis="y", alpha=0.2)

    for idx, (label, _, scan_work) in enumerate(cum_data):  # cum_data: cumulative scan_work
        cx2.plot(x, scan_work, color=colors[idx % len(colors)], linewidth=2.4,
                  marker="o", markersize=5, label=label)
        final = scan_work[-1]
        pct = (baseline_final2 - final) / baseline_final2 * 100.0 if baseline_final2 else 0.0
        text = f"{final:.1f}s" if idx == 0 else f"{final:.1f}s ({pct:+.1f}%)"
        cx2.annotate(text, xy=(x[-1], final), xytext=(x[-1] + 0.5, label_ys2[idx]),
                     textcoords="data", ha="left", va="center", fontsize=12, color=ink_primary)
    cx2.set_ylim(top=max(finals2) * 1.14)
    cx2.set_xlabel("TPC-H query", fontsize=18, color=ink_primary)
    cx2.set_ylabel("Cumulative scan work (s)", fontsize=18, color=ink_primary)
    cx2.tick_params(axis="y", colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    cx2.tick_params(axis="x", colors=ink_primary, labelcolor=ink_primary)
    cx2.set_xticks(x)
    cx2.set_xticklabels(x_labels, fontsize=10, color=ink_primary, rotation=90)
    cx2.set_xlim(0, n + 3)
    cx2.grid(axis="y", alpha=0.2)

    handles3, labels3 = cx1.get_legend_handles_labels()
    fig3.legend(handles3, labels3, frameon=True, edgecolor=ink_primary, fontsize=17,
                loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=len(stage_data),
                labelcolor=ink_primary)

    fig3.tight_layout()
    out_png3 = REPO_ROOT / "experiment" / "graph" / "random22_sf50_3stage_perquery_hitrate_cumulative_scanwork.png"
    fig3.savefig(out_png3, dpi=200, bbox_inches="tight")
    print(f"wrote {out_png3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
