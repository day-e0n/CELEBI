#!/usr/bin/env python3
"""Where the GPU baseline's time goes, next to DuckDB on the CPU.

Both workloads as in plot_headline_cache_reorder.py, page cache off, prefetch off, executions
2-3 averaged:
  GPU (O_DIRECT)     the shipped read path: every scan reads the parquet file from NVMe
  GPU (page cache)   BENCH_USE_ODIRECT=0: reads go through the kernel page cache, so from
                     execution 2 on the compressed bytes come from host RAM
  CPU 40 / 10 thr.   DuckDB alone (scripts/run_cpu_baseline.py); 10 threads = one GPU's share
                     of this 40-thread, 4-GPU box. DuckDB also reads through the page cache.

GPU bars are split by what the query's tasks were doing (task telemetry):
  disk wait     scan time that disappears when the bytes come from RAM
                (scan time with O_DIRECT minus scan time through the page cache)
  scan          time with >= 1 GPU_SCAN running, reading from RAM: copy, decompress, decode
  other ops     joins, aggregations, ... with no scan running
  rest          no task computing: planning, result fetch, gaps between pipelines

  pixi run -e duckdb-python python result/plot_gpu_vs_cpu_breakdown.py
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

EXP = ROOT / "experiment"
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("Q", OUT / "tpch_sf50_per_query.py")

WORKLOADS = [
    ("TPC-H SF50", ["RO_worst_base", "RO_worst_base_r2", "RO_worst_base_r3"], ["OD0_th_base"],
     "CPU_th_t40", "CPU_th_t10"),
    ("ClickBench", ["F_cb_off_base"], ["OD0_cb_base"], "CPU_cb_t40", "CPU_cb_t10"),
]
COLORS = {"disk wait": "#b4522e", "scan": "#3d6b66", "other ops": "#9fb8b4", "rest": "#d9dcdb"}


def wall(run: str) -> float:
    per_exec: dict[int, float] = {}
    for line in open(EXP / run / "runner.log"):
        m = re.search(r"exec=(\d+)/\d+ .* done wall=([\d.]+)s", line)
        if m and int(m[1]) >= 2:
            per_exec[int(m[1])] = per_exec.get(int(m[1]), 0.0) + float(m[2])
    return sum(per_exec.values()) / len(per_exec)


def gpu_parts(runs: list[str]) -> dict[str, float]:
    """Mean over runs of wall, scan-running time and other-operator time (no scan)."""
    # Reuse the task-telemetry sweep from tpch_sf50_per_query.py without running its main body.
    src = (OUT / "tpch_sf50_per_query.py").read_text().split("FIELDS = ")[0]
    ns: dict = {"__file__": str(OUT / "tpch_sf50_per_query.py")}
    exec(compile(src, "tpch_sf50_per_query.py", "exec"), ns)
    tot = {"wall": 0.0, "scan": 0.0, "other": 0.0}
    for run in runs:
        spans = ns["task_spans"](run)
        n_exec = len({e for _, e in spans})
        scan = other = 0.0
        for s in spans.values():
            p = ns["split"](s)
            scan += p["scan_full"] + p["scan_resv"] + p["scan_idle"]
            other += p["other_ops"]
        tot["wall"] += wall(run) / len(runs)
        tot["scan"] += scan / n_exec / len(runs)
        tot["other"] += other / n_exec / len(runs)
    return tot


def main() -> int:
    fig, axes = plt.subplots(1, len(WORKLOADS), figsize=(6.6, 2.8))
    for ax, (name, odirect, pagecache, cpu40, cpu10) in zip(axes, WORKLOADS):
        od, pc = gpu_parts(odirect), gpu_parts(pagecache)
        bars = [
            ("GPU\n(O_DIRECT)", {"disk wait": od["scan"] - pc["scan"], "scan": pc["scan"],
                                 "other ops": od["other"],
                                 "rest": od["wall"] - od["scan"] - od["other"]}),
            ("GPU\n(page cache)", {"disk wait": 0.0, "scan": pc["scan"], "other ops": pc["other"],
                                   "rest": pc["wall"] - pc["scan"] - pc["other"]}),
            ("CPU\n40 thr.", {"total": wall(cpu40)}),
            ("CPU\n10 thr.", {"total": wall(cpu10)}),
        ]
        for x, (label, parts) in enumerate(bars):
            bottom = 0.0
            if "total" in parts:
                ax.bar(x, parts["total"], 0.6, zorder=3, **ps.NEUTRAL)
                bottom = parts["total"]
            else:
                for key in ("disk wait", "scan", "other ops", "rest"):
                    v = max(parts[key], 0.0)
                    ax.bar(x, v, 0.6, bottom=bottom, color=COLORS[key], edgecolor="#333333",
                           linewidth=0.5, zorder=3, label=key if x == 0 else None)
                    bottom += v
            ax.text(x, bottom + 0.6, f"{bottom:.1f}s", ha="center", va="bottom", fontsize=6.5)
            print(f"{name:11} {label.replace(chr(10), ' '):18} {bottom:6.2f}s  "
                  + " ".join(f"{k}={v:.2f}" for k, v in parts.items()))
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([b[0] for b in bars], fontsize=7)
        ax.set_ylim(0, max(od["wall"], wall(cpu40), wall(cpu10)) * 1.25)
        ps.finish(ax, "query time (s)" if ax is axes[0] else None)
        ax.set_title(name, fontsize=8.5, pad=4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"gpu_vs_cpu_breakdown.{ext}", dpi=300)
    print(OUT / "gpu_vs_cpu_breakdown.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
