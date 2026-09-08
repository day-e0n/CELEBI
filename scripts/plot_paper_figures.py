#!/usr/bin/env python3
"""Every paper figure for the page-cache work, in one pass.

  fig1_scan_latency     absolute scan latency per condition, both workloads
  fig2_reduction        the same normalised to the uncached baseline
  fig3_cache_key        what re-keying the cache by (file, filter) is worth
  fig4_reorder_cost     per-query cost of the reorder on TPC-H SF100

`fixed` caches only fixed-width columns; STRING columns are read from parquet.
`fixed+var` adds them as variable-width pages. Each cache condition runs twice --
in the queries' arrival order and after the cache-aware reorder -- so the cache's
contribution and the reorder's are separable. Arrival order is the least
favourable one: adjacent queries share as few columns as possible.

Hot scan latency throughout; execution 1 of every run is the cold pass and is
dropped, and error bars are the spread over 3 repeats.

Usage:  pixi run -e duckdb-python python scripts/plot_paper_figures.py
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from paper_style import (  # noqa: E402
    ACCENT, DARK, ERRBAR, LIGHT, NEUTRAL, finish, plt, save, scan_seconds, series)

CB = REPO / "experiment/cb_worst"
TP = REPO / "experiment/tp100_worst"
WORKLOADS = [("ClickBench", CB), ("TPC-H SF100", TP)]
GROUPS = [("fixed", "nr_fixed", "fixed"), ("fixed+var", "nr_fx_var", "fx_var")]


def fig_scan_latency() -> None:
    """Absolute seconds -- keeps the two workloads' different scales visible."""
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3))
    for ax, (title, root) in zip(axes, WORKLOADS):
        base = series(root, "base")
        xs, heights, errs, styles = [0.0], [base[0]], [base[1]], [NEUTRAL]
        ticks, labels = [0.0], ["none"]
        x = 1.0
        for label, plain, reord in GROUPS:
            for off, tag, style in ((-0.19, plain, LIGHT), (0.19, reord, DARK)):
                s = series(root, tag)
                if s is None:
                    continue
                xs.append(x + off)
                heights.append(s[0])
                errs.append(s[1])
                styles.append(style)
            ticks.append(x)
            labels.append(label)
            x += 1.0
        for xi, h, e, st in zip(xs, heights, errs, styles):
            ax.bar(xi, h, 0.36, yerr=e, capsize=1.8, error_kw=ERRBAR, zorder=3, **st)
            ax.text(xi, h + e + max(heights) * 0.02, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.6, x - 0.4)
        ax.set_ylim(0, max(heights) * 1.18)
        ax.set_title(title, fontsize=9, pad=5)
        finish(ax)
    axes[0].set_ylabel("Scan latency (s)", fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, **s) for s in (NEUTRAL, LIGHT, DARK)]
    fig.legend(handles, ["no cache", "arrival order", "+ reorder"],
               loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3,
               fontsize=8, handlelength=1.5, handleheight=0.95, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig1_scan_latency")


def fig_reduction() -> None:
    """Normalised: how much of the uncached scan each configuration removes."""
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    pos, ticks, labels, spans = [], [], [], []
    x = 0.0
    for title, root in WORKLOADS:
        start = x
        base = series(root, "base")[0]
        for label, plain, reord in GROUPS:
            for off, tag, style in ((-0.19, plain, LIGHT), (0.19, reord, DARK)):
                s = series(root, tag)
                if s is None:
                    continue
                pct, err = (1 - s[0] / base) * 100, s[1] / base * 100
                ax.bar(x + off, pct, 0.36, yerr=err, capsize=1.8,
                       error_kw=ERRBAR, zorder=3, **style)
                ax.text(x + off, pct + err + 0.9, f"{pct:.0f}",
                        ha="center", va="bottom", fontsize=7)
            ticks.append(x)
            labels.append(label)
            x += 1.0
        spans.append((f"{title}  (baseline {base:.0f} s)", start, x - 1.0))
        x += 0.7
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.65, x - 1.35)
    ax.set_ylim(0, 48)
    for title, lo, hi in spans:
        ax.text((lo + hi) / 2, -7.5, title, ha="center", va="top", fontsize=8.5)
    finish(ax, "Scan latency reduction (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, **s) for s in (LIGHT, DARK)]
    ax.legend(handles, ["arrival order", "+ cache-aware reorder"],
              loc="upper left", fontsize=8, handlelength=1.5, handleheight=0.95,
              borderaxespad=0.2, labelspacing=0.3)
    fig.tight_layout()
    save(fig, "fig2_reduction")


def fig_cache_key() -> None:
    """Keying the cache by (file, filter) instead of by projection, ClickBench."""
    colkey = REPO / "experiment/clickbench_colkey"
    widen = REPO / "experiment/clickbench_widen"
    bars = [("no cache", series(colkey, "base"), NEUTRAL),
            ("keyed by\nprojection", series(colkey, "proj"), LIGHT),
            ("keyed by\n(file, filter)", series(widen, "ck6"), DARK)]
    fig, ax = plt.subplots(figsize=(2.9, 2.4))
    for i, (label, s, style) in enumerate(bars):
        ax.bar(i, s[0], 0.5, yerr=s[1], capsize=1.8, error_kw=ERRBAR, zorder=3, **style)
        ax.text(i, s[0] + s[1] + 1.0, f"{s[0]:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylim(0, max(b[1][0] for b in bars) * 1.18)
    ax.set_title("ClickBench: cache identity", fontsize=9, pad=5)
    finish(ax, "Scan latency (s)")
    fig.tight_layout()
    save(fig, "fig3_cache_key")


def fig_reorder_cost() -> None:
    """Where the reorder's regression on TPC-H SF100 comes from, per query."""
    def per_q(tag):
        out = defaultdict(list)
        for r in (1, 2, 3):
            p = TP / f"{tag}_{r}" / "bucket.csv"
            if not p.exists():
                continue
            rows = [x for x in csv.DictReader(p.open()) if int(x["execution"]) > 1]
            n = len({int(x["execution"]) for x in rows})
            acc = defaultdict(float)
            for x in rows:
                acc[x["query"]] += float(x["scan"])
            for q, v in acc.items():
                out[q].append(v / n)
        return {q: statistics.mean(v) for q, v in out.items()}

    a, b = per_q("nr_fx_var"), per_q("fx_var")
    deltas = sorted(((b[q] - a[q]) / 1000.0, q) for q in a if q in b)
    deltas = [d for d in deltas if abs(d[0]) >= 0.2]

    fig, ax = plt.subplots(figsize=(5.6, 2.1))
    ys = np.arange(len(deltas))
    for y, (d, q) in zip(ys, deltas):
        ax.barh(y, d, 0.62, zorder=3, **(ACCENT if d > 0 else DARK))
        ax.text(d + (0.12 if d > 0 else -0.12), y, f"{d:+.1f}",
                va="center", ha="left" if d > 0 else "right", fontsize=7)
    ax.set_yticks(ys)
    ax.set_yticklabels([q for _, q in deltas], fontsize=8)
    ax.axvline(0, color="#2f3b3a", linewidth=0.6, zorder=4)
    ax.set_xlabel("Change in scan latency from reordering (s)", fontsize=8.5)
    # Leave room on the left for the negative bars' labels, which would otherwise
    # collide with the query names on the axis.
    ax.set_xlim(min(d for d, _ in deltas) * 2.6, max(d for d, _ in deltas) * 1.22)
    ax.xaxis.grid(True, color="0.88", linewidth=0.5, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_title("TPC-H SF100, fixed+var: the reorder moves q21 and q12 to the front",
                 fontsize=8.5, pad=5, loc="left")
    fig.tight_layout()
    save(fig, "fig4_reorder_cost")


if __name__ == "__main__":
    fig_scan_latency()
    fig_reduction()
    fig_cache_key()
    fig_reorder_cost()
