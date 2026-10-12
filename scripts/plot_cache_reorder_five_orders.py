#!/usr/bin/env python3
"""What the page cache and small-first-big-last are worth, over five arrival orders.

The reorder's headline number used to come from ClickBench's natural arrival order
alone. Running four more orders puts it between -33.7% and +50.7%, so the left panel
plots every order rather than a mean: the cache is a consistent win, the reorder is
not. TPC-H sits beside it at the same scale, where the policy provably changes no
order at all (lineitem.l_comment is the only column over budget and no query in the
reorder's column table reads it).

Every bar is the mean of executions 2 and 3; execution 1 is the cold pass.

  pixi run -e duckdb-python python scripts/plot_cache_reorder_five_orders.py
"""
from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "experiment"

# ClickBench: one pair of runs per arrival order, cache on both sides.
CB_ORDERS = [("natural", "F_cb_cache", "F_cb_reorder"),
             ("worst_case", "S_cb_worst_c", "S_cb_worst_r"),
             ("random_a", "T_cb_randa_c", "T_cb_randa_r"),
             ("random_b", "T_cb_randb_c", "T_cb_randb_r"),
             ("random_c", "T_cb_randc_c", "T_cb_randc_r")]
CB_BASE = "F_cb_base"          # natural order, cache off
TH = [("baseline", "F_th_base"), ("cache", "F_th_cache"), ("cache\n+reorder", "X_th_reord_1")]
TH_STYLES = [ps.NEUTRAL, ps.LIGHT, ps.DARK]


def warm_total(run: str) -> float:
    """Summed GPU operator seconds for one warm execution, averaged over 2 and 3."""
    per: dict[int, float] = collections.defaultdict(float)
    with (ROOT / run / "bucket.csv").open() as handle:
        for row in csv.DictReader(handle):
            execution = int(row["execution"])
            if execution == 1:
                continue
            per[execution] += float(row["total_ms"]) / 1000.0
    if not per:
        raise SystemExit(f"{run}: no warm executions")
    return sum(per.values()) / len(per)


def main() -> None:
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[5, 2], wspace=0.28))

    # --- ClickBench -------------------------------------------------------
    x = np.arange(len(CB_ORDERS))
    plain = [warm_total(c) for _, c, _ in CB_ORDERS]
    reord = [warm_total(r) for _, _, r in CB_ORDERS]
    width = 0.36
    left.bar(x - width / 2, plain, width, label="cache on", zorder=3, **ps.LIGHT)
    left.bar(x + width / 2, reord, width, label="cache on + reorder", zorder=3, **ps.DARK)

    base = warm_total(CB_BASE)
    left.axhline(base, color=ps.ACCENT["facecolor"], linewidth=0.9, linestyle="--", zorder=4)
    left.text(-0.42, base - 6, f"no cache, natural order ({base:.0f}s)",
              ha="left", va="top", fontsize=7.5, color=ps.ACCENT["facecolor"])

    # The delta sits above whichever bar of the pair is taller; a bar that reaches the
    # baseline rule pushes its label above the rule instead of onto it.
    for xi, (before, after) in enumerate(zip(plain, reord)):
        delta = (after - before) / before * 100.0
        top = max(before, after)
        y = base + 8 if top > base - 22 else top + 5
        left.text(xi, y, f"{delta:+.0f}%", ha="center", va="bottom",
                  fontsize=8, color="#b4522e" if delta > 0 else "#2f3b3a")

    left.set_xticks(x)
    left.set_xticklabels([name for name, _, _ in CB_ORDERS])
    left.set_xlabel("ClickBench 43 queries — arrival order", fontsize=8.5, labelpad=3)
    left.set_ylim(0, base * 1.30)
    left.legend(loc="upper left", fontsize=7.5, ncol=2, handlelength=1.2,
                bbox_to_anchor=(0.0, 0.93))
    ps.finish(left, "GPU operator time (s)")

    # --- TPC-H ------------------------------------------------------------
    xt = np.arange(len(TH))
    vals = [warm_total(run) for _, run in TH]
    for xi, value, style in zip(xt, vals, TH_STYLES):
        right.bar(xi, value, 0.55, zorder=3, **style)
        right.text(xi, value + 0.4, f"{value:.1f}", ha="center", va="bottom", fontsize=7.5)

    # The policy returns TPC-H's input order unchanged, so the third bar is the second
    # one measured again -- the gap between them is this benchmark's run-to-run spread.
    # The reorder's cost here is a warm-up pass, not throughput; see
    # scripts/plot_tpch_reorder_convergence.py.
    rule = max(vals[1], vals[2]) + 4.0
    right.annotate("", xy=(2, rule), xytext=(1, rule),
                   arrowprops=dict(arrowstyle="<->", linewidth=0.6, color="#2f3b3a"))
    right.text(1.5, rule + 0.6, "+5.3%", ha="center", va="bottom",
               fontsize=7.5, color="#b4522e", linespacing=1.1)

    right.set_xticks(xt)
    right.set_xticklabels([name for name, _ in TH])
    right.set_xlabel("TPC-H SF50 22 queries", fontsize=8.5, labelpad=3)
    right.set_ylim(0, max(vals) * 1.42)
    ps.finish(right, None)

    ps.save(fig, "cache_reorder_five_orders")


if __name__ == "__main__":
    main()
