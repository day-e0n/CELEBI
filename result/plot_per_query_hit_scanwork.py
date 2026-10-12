#!/usr/bin/env python3
"""Per query: cache hit ratio (left) and cumulative scan work (right), q1..qN in query-number
order, for baseline / paging / paging + reorder. Same layout as
scripts/plot_hitrate_and_scanwork_combined.py.

  hit ratio   bytes the page cache served / decoded bytes the query's scans handed downstream
              (SIRIUS_CACHE_HIT_TRACE=1, experiment/HIT2_*, 1 run, executions 2-3)
  scan work   seconds GPU_SCAN tasks spent computing, summed over the query's scan tasks
              (task telemetry of experiment/A_* -- TPC-H 3 runs, ClickBench 2 -- executions 2-3).
              Up to four scan tasks run at once, so this adds up past wall time.
Each query's own value is shown at its query number wherever it ran in the reordered batch.
Current engine: late materialization on, scan no-history reservation 3x. "paging + reorder" is
the reordered batch with order-aware eviction.

  pixi run -e duckdb-python python result/plot_per_query_hit_scanwork.py
writes per_query_hit_scanwork_{tpch,clickbench}.{png,pdf,csv}
"""
from __future__ import annotations

import collections
import csv
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import cache_hit_rate as chr_  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42

_src = (OUT / "tpch_sf50_per_query.py").read_text().split("FIELDS = ")[0]
_ns: dict = {"__file__": str(OUT / "tpch_sf50_per_query.py")}
exec(compile(_src, "tpch_sf50_per_query.py", "exec"), _ns)

CONDS = [("Baseline", "base", "#4c78a8"), ("Paging", "lru", "#59a14f"),
         ("Paging + reorder", "best", "#f2b134")]
WORKLOADS = [("tpch", "TPC-H", "th", (1, 2, 3), [q for q in range(1, 23)]),
             ("clickbench", "ClickBench", "cb", (1, 2), [q for q in range(1, 44) if q not in (28, 29)])]
INK = "#0b0b0b"


def hit_by_query(bench: str, cond: str) -> dict[int, float]:
    data = chr_.collect(f"HIT2_{bench}_{cond}")
    out = {}
    for q in {q for q, _ in data}:
        s = chr_.summarize(c for (q2, _), c in data.items() if q2 == q)
        out[q] = s["byte_hit_pct"]
    return out


def scan_work_by_query(run: str) -> dict[int, float]:
    """Per-execution GPU_SCAN computing seconds per query (executions 2-3 averaged)."""
    spans = _ns["task_spans"](run)
    per = collections.defaultdict(float)
    execs = {e for _, e in spans}
    for (q, e), s in spans.items():
        per[q] += sum((b - a) / 1e9 for a, b, k in s if k == "scan")
    return {q: v / len(execs) for q, v in per.items()}


def main() -> int:
    for slug, title, bench, reps, queries in WORKLOADS:
        rows = []
        series = []
        for label, cond, color in CONDS:
            hit = hit_by_query(bench, cond)
            works = [scan_work_by_query(f"A_{bench}_{cond}_r{r}") for r in reps]
            work = {q: statistics.mean(w.get(q, 0.0) for w in works) for q in queries}
            hits = [hit.get(q, 0.0) for q in queries]
            cum, total = [], 0.0
            for q in queries:
                total += work[q]
                cum.append(total)
            series.append((label, color, hits, cum))
            for q, h in zip(queries, hits):
                rows.append({"condition": label, "query": f"q{q}", "hit_pct": round(h, 1),
                             "scan_work_s": round(work[q], 3)})
        with open(OUT / f"per_query_hit_scanwork_{slug}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        n = len(queries)
        xs = list(range(1, n + 1))
        labels = [f"q{q}" for q in queries]
        wide = n > 25
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5 if wide else 12.5, 4.6))
        for label, color, hits, cum in series:
            ax1.plot(xs, hits, color=color, linewidth=2.0, marker="o", markersize=4, label=label)
        ax1.set_ylabel("Per-query cache hit ratio\n(% of scanned bytes)", fontsize=14, color=INK)
        ax1.set_ylim(-2, 102)
        base_final = series[0][3][-1]
        for i, (label, color, hits, cum) in enumerate(series):
            ax2.plot(xs, cum, color=color, linewidth=2.0, marker="o", markersize=4, label=label)
            final = cum[-1]
            text = f"{final:.1f}s" if i == 0 else f"{final:.1f}s ({100 * (final / base_final - 1):+.1f}%)"
            ax2.annotate(text, (xs[-1], final), textcoords="offset points",
                         xytext=(8, (6 - 9 * i)), ha="left", va="center", fontsize=10, color=INK)
        ax2.set_ylabel("Cumulative scan work (s)", fontsize=14, color=INK)
        ax2.set_ylim(0, None)
        for ax in (ax1, ax2):
            ax.set_xlabel(f"{title} query", fontsize=14, color=INK)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=90, fontsize=8 if wide else 10, color=INK)
            ax.tick_params(axis="y", labelsize=12, colors=INK)
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        ax1.set_xlim(0.3, n + 0.7)
        ax2.set_xlim(0.3, n + (5 if wide else 3.5))
        handles, leg = ax1.get_legend_handles_labels()
        fig.legend(handles, leg, frameon=True, edgecolor=INK, fontsize=13, loc="upper center",
                   bbox_to_anchor=(0.5, 1.07), ncol=3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"per_query_hit_scanwork_{slug}.{ext}", dpi=200, bbox_inches="tight")
        print(title, "final scan work:", " ".join(f"{s[0]}={s[3][-1]:.1f}s" for s in series))
        print(OUT / f"per_query_hit_scanwork_{slug}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
