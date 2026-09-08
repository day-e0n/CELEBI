#!/usr/bin/env python3
"""Figures for the SF100 reordering study.

Four panels, each answering one question the measurements settled:

  fig_r1  which order is fastest, over every order we ran
  fig_r2  what the overlap reorder actually buys (variance, not mean)
  fig_r3  what predicts total time (hits do; entry width does not)
  fig_r4  the dynamic-filter gate, all three directions

Totals rather than scan-only: the reorder moves whole queries, so join and
aggregate time moves with them and a scan-only axis would hide that.
"""
from __future__ import annotations

import csv
import glob
import math
import re
import statistics
from pathlib import Path

import paper_style as ps
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def totals(pattern: str) -> list[float]:
    """One number per warm execution: the whole query set's GPU time, seconds."""
    out: list[float] = []
    for path in sorted(glob.glob(str(ROOT / pattern))):
        per_exec: dict[str, float] = {}
        for row in csv.DictReader(open(path)):
            if row["execution"] == "1":      # warmup
                continue
            per_exec.setdefault(row["execution"], 0.0)
            per_exec[row["execution"]] += float(row["total_ms"])
        out += [v / 1000.0 for v in per_exec.values()]
    return out


def stat(pattern: str) -> tuple[float, float, int]:
    v = totals(pattern)
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0), len(v)


ORDERS = [
    ("No cache",            "experiment/tp100_worst/base_*/bucket.csv",              ps.NEUTRAL),
    ("shared-per-cost",     "experiment/tp100_sharedcost/sharedcost_*/bucket.csv",   ps.LIGHT),
    ("q1 first",            "experiment/tp100_q1first/q1first_*/bucket.csv",         ps.LIGHT),
    ("Overlap reorder",     "experiment/tp100_random/reo_*/bucket.csv",              ps.ACCENT),
    ("Random arrival",      "experiment/tp100_random/arr_*/bucket.csv",              ps.NEUTRAL),
    ("cost-asc (estimate)", "experiment/tp100_policy/pol_*/bucket.csv",              ps.LIGHT),
    ("cost-desc",           "experiment/tp100_desc/desc_*/bucket.csv",               ps.LIGHT),
    ("byte-overlap",        "experiment/tp100_bov/bov_*/bucket.csv",                 ps.LIGHT),
    ("widest-first",        "experiment/tp100_wf/wf_*/bucket.csv",                   ps.LIGHT),
    ("shared-first",        "experiment/tp100_shared/shared_*/bucket.csv",           ps.LIGHT),
    ("cost-asc (measured)", "experiment/tp100_asc/asc_*/bucket.csv",                 ps.DARK),
]


def fig_orders() -> None:
    rows = []
    for label, pattern, style in ORDERS:
        mean, sd, n = stat(pattern)
        rows.append((mean, sd, n, label, style))
    rows.sort(reverse=True)

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    y = range(len(rows))
    ax.barh(list(y), [r[0] for r in rows], height=0.66,
            xerr=[r[1] for r in rows], error_kw=ps.ERRBAR, zorder=3,
            **{k: [r[4][k] for r in rows] for k in ("facecolor", "edgecolor")},
            linewidth=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[3] for r in rows])
    for i, (mean, sd, n, label, _) in enumerate(rows):
        ax.text(mean + sd + 2.5, i, f"{mean:.1f}", va="center", fontsize=7.2)
    ax.set_xlabel("workload GPU time (s)", fontsize=8.5)
    ax.set_xlim(0, 245)
    ax.xaxis.grid(True, color=ps.GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(pad=2)
    ps.save(fig, "fig_r1_orders")
    plt.close(fig)


def fig_variance() -> None:
    """Per arrival order, before and after the overlap reorder."""
    pairs = []
    for i in (1, 2, 3, 4):
        a, _, _ = stat(f"experiment/tp100_random/arr_{i}/bucket.csv")
        b, _, _ = stat(f"experiment/tp100_random/reo_{i}/bucket.csv")
        pairs.append((f"R{i}", a, b))

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    xs = range(len(pairs))
    width = 0.36
    ax.bar([x - width / 2 for x in xs], [p[1] for p in pairs], width,
           zorder=3, label="arrival order", **ps.NEUTRAL)
    ax.bar([x + width / 2 for x in xs], [p[2] for p in pairs], width,
           zorder=3, label="after overlap reorder", **ps.ACCENT)
    for x, (_, a, b) in zip(xs, pairs):
        ax.annotate(f"{(a - b) / a * 100:+.0f}%", (x + width / 2, b + 3),
                    ha="center", fontsize=7, color="#b4522e")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([p[0] for p in pairs])
    ax.set_ylim(0, 230)
    ax.legend(fontsize=7.2, loc="lower right", ncol=1)
    ps.finish(ax, "workload GPU time (s)")
    ps.save(fig, "fig_r2_variance")
    plt.close(fig)


FEATURES = [
    ("겹침재정렬",  "experiment/tp100_random/reo_1",            "experiment/tp100_random/reo_*/bucket.csv"),
    ("q1-first",   "experiment/tp100_q1first/q1first_1",       "experiment/tp100_q1first/q1first_*/bucket.csv"),
    ("shared/cost","experiment/tp100_sharedcost/sharedcost_1", "experiment/tp100_sharedcost/sharedcost_*/bucket.csv"),
    ("cost-desc",  "experiment/tp100_desc/desc_1",             "experiment/tp100_desc/desc_*/bucket.csv"),
    ("widest",     "experiment/tp100_wf/wf_1",                 "experiment/tp100_wf/wf_*/bucket.csv"),
    ("byte-ovl",   "experiment/tp100_bov/bov_1",               "experiment/tp100_bov/bov_*/bucket.csv"),
    ("shared",     "experiment/tp100_shared/shared_1",         "experiment/tp100_shared/shared_*/bucket.csv"),
    ("cost-asc-e", "experiment/tp100_policy/pol_1",            "experiment/tp100_policy/pol_*/bucket.csv"),
    ("cost-asc",   "experiment/tp100_asc/asc_1",               "experiment/tp100_asc/asc_*/bucket.csv"),
    ("worst-arr",  "experiment/tp100_worst/nr_fx_var_1",       "experiment/tp100_worst/nr_fx_var_*/bucket.csv"),
] + [(f"rand{i}", f"experiment/tp100_random/arr_{i}",
      f"experiment/tp100_random/arr_{i}/bucket.csv") for i in (1, 2, 3, 4)]


def log_features(run_dir: str) -> dict | None:
    logs = glob.glob(str(ROOT / run_dir / "celebi_fixed_variable" / "log_dir" / "*.log"))
    if not logs:
        return None
    text = open(logs[0]).read()
    m = re.search(r"populate_direct table='[^']*lineitem\.parquet;' rows=\d+ columns=(\d+)", text)
    return {"hits": text.count("] using"),
            "width": int(m.group(1)) if m else 0}


def pearson(a, b) -> float:
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0


def fig_predictors() -> None:
    """Cache hits track total time; the entry's opening width does not."""
    hits, widths, times = [], [], []
    for _, run_dir, pattern in FEATURES:
        f = log_features(run_dir)
        if not f:
            continue
        hits.append(f["hits"])
        widths.append(f["width"])
        times.append(stat(pattern)[0])

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.8), sharey=True)
    for ax, xs, xlabel in ((axes[0], hits, "cache hits per run"),
                           (axes[1], widths, "columns in first lineitem entry")):
        ax.scatter(xs, times, s=13, facecolor=ps.DARK["facecolor"],
                   edgecolor=ps.DARK["edgecolor"], linewidth=0.5, zorder=3)
        ax.set_xlabel(xlabel, fontsize=7.6)
        ax.annotate(f"r = {pearson(xs, times):+.2f}", (0.96, 0.93),
                    xycoords="axes fraction", ha="right", fontsize=7.6)
        ps.finish(ax)
    axes[0].set_ylabel("workload GPU time (s)", fontsize=8.5)
    fig.subplots_adjust(wspace=0.12)
    ps.save(fig, "fig_r3_predictors")
    plt.close(fig)


def fig_dynamic_filter() -> None:
    """Both gate directions cost time, for different reasons."""
    labels = ["gates shut\n(baseline)", "read gate open"]
    patterns = ["experiment/tp100_worst/nr_fx_var_*/bucket.csv",
                "experiment/tp100_dfread/dfr_*/bucket.csv"]
    means = [stat(p)[0] for p in patterns]

    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    xs = list(range(len(labels) + 1))
    ax.bar(xs[:2], means, 0.55, zorder=3,
           **{k: [ps.NEUTRAL[k], ps.ACCENT[k]] for k in ("facecolor", "edgecolor")},
           linewidth=0.6)
    # Both gates open never produced a number: it aborted at the same query in
    # 3 of 3 runs, so it is drawn as an outline rather than left off the axis.
    ax.bar([xs[2]], [means[0]], 0.55, zorder=3, facecolor="none",
           edgecolor=ps.ACCENT["edgecolor"], linewidth=0.6, linestyle=(0, (2, 1.6)))
    ax.text(xs[2], means[0] / 2, "OOM\n3 of 3", ha="center", va="center", fontsize=7.2)
    ax.annotate(f"{(means[1] - means[0]) / means[0] * 100:+.1f}%",
                (xs[1], means[1] + 4), ha="center", fontsize=7.2, color="#b4522e")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels + ["both gates open"], fontsize=7.6)
    ax.set_ylim(0, 215)
    ps.finish(ax, "workload GPU time (s)")
    ps.save(fig, "fig_r4_dynamic_filter")
    plt.close(fig)


if __name__ == "__main__":
    fig_orders()
    fig_variance()
    fig_predictors()
    fig_dynamic_filter()
