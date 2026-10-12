#!/usr/bin/env python3
"""Where the time goes now, per condition: baseline / paging / paging + reorder / + order-aware.

Each condition was run twice: with O_DIRECT (the shipped read path; experiment/FA_*) and with the
parquet bytes in host RAM (BENCH_USE_ODIRECT=0; experiment/RAM2_*). The difference is the disk
wait that condition still pays. The RAM run's own time is split by task telemetry into:
  scan       >= 1 GPU_SCAN running: host->GPU copy, decompress, decode (and cache serving)
  other ops  joins, aggregations, ... with no scan running
  rest       nothing computing: planning, result fetch, gaps
Prefetch off, executions 2-3. TPC-H SF50 4 GB worst order; ClickBench 6 GB natural order.

  pixi run -e duckdb-python python result/plot_breakdown_now.py
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
EXP = ROOT / "experiment"
sys.path.insert(0, str(ROOT / "scripts"))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Reuse the task-telemetry sweep from tpch_sf50_per_query.py without running its main body.
_src = (OUT / "tpch_sf50_per_query.py").read_text().split("FIELDS = ")[0]
_ns: dict = {"__file__": str(OUT / "tpch_sf50_per_query.py")}
exec(compile(_src, "tpch_sf50_per_query.py", "exec"), _ns)

# "paging + reorder" is the reordered batch with order-aware eviction.
CONDS = [("baseline", "base"), ("paging", "lru"), ("paging\n+ reorder", "ro_evict")]
WORKLOADS = [("TPC-H SF50", "th", (1, 2, 3)), ("ClickBench", "cb", (1, 2))]
COLORS = {"disk wait": "#b4522e", "scan": "#3d6b66", "other ops": "#9fb8b4", "rest": "#d9dcdb"}


def wall(run: str) -> float:
    per_exec: dict[int, float] = {}
    for line in open(EXP / run / "runner.log"):
        m = re.search(r"exec=(\d+)/\d+ .* done wall=([\d.]+)s", line)
        if m and int(m[1]) >= 2:
            per_exec[int(m[1])] = per_exec.get(int(m[1]), 0.0) + float(m[2])
    return sum(per_exec.values()) / len(per_exec)


def parts(run: str) -> dict[str, float]:
    spans = _ns["task_spans"](run)
    n_exec = len({e for _, e in spans}) or 1
    scan = other = 0.0
    for s in spans.values():
        p = _ns["split"](s)
        scan += p["scan_full"] + p["scan_resv"] + p["scan_idle"]
        other += p["other_ops"]
    w = wall(run)
    return {"wall": w, "scan": scan / n_exec, "other ops": other / n_exec,
            "rest": w - scan / n_exec - other / n_exec}


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, (title, bench, reps) in zip(axes, WORKLOADS):
        for x, (label, cond) in enumerate(CONDS):
            od = statistics.mean(wall(f"FA_{bench}_{cond}_r{r}") for r in reps)
            ram = parts(f"RAM2_{bench}_{cond}")
            segs = {"disk wait": max(od - ram["wall"], 0.0), "scan": ram["scan"],
                    "other ops": ram["other ops"], "rest": max(ram["rest"], 0.0)}
            bottom = 0.0
            for key in ("disk wait", "scan", "other ops", "rest"):
                ax.bar(x, segs[key], 0.6, bottom=bottom, color=COLORS[key], edgecolor="#333333",
                       linewidth=0.5, zorder=3, label=key if x == 0 else None)
                bottom += segs[key]
            ax.text(x, bottom + 0.5, f"{od:.1f}s", ha="center", va="bottom", fontsize=6.5)
            print(f"{title:11} {label.replace(chr(10), ' '):32} O_DIRECT {od:6.2f}s  RAM {ram['wall']:6.2f}s  "
                  + "  ".join(f"{k}={v:.2f}" for k, v in segs.items()))
        ax.set_xticks(range(len(CONDS)))
        ax.set_xticklabels([c[0] for c in CONDS], fontsize=6.3)
        ax.set_ylim(0, 40)
        ps.finish(ax, "query time (s)" if ax is axes[0] else None)
        ax.set_title(title, fontsize=8.5, pad=4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"breakdown_now.{ext}", dpi=300)
    print(OUT / "breakdown_now.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
