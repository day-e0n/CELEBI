#!/usr/bin/env python3
"""The scan path only, from a query's start to the batch the scan hands downstream.

Per execution (executions 2-3), for baseline / paging / paging + reorder / + order-aware:
  query setup      wall time outside every task (planning, result fetch)        [wall]
  -- summed over scan tasks (up to 4 run at once, so these add up past wall) --
  reserve wait     scan tasks in Reserving (waiting for a GPU memory reservation)
  (scan tasks also wait in a queue for a worker slot -- created ahead of time, 177-195s summed
   on TPC-H -- which only reflects that there is more work than slots; left out)
  disk wait        read time with O_DIRECT minus the same with the bytes in RAM
  decode           read time with the bytes in RAM: host->GPU copy, decompress, decode
  cache serve      cache lookup, page gathering, splicing cached and read columns
  cache populate   copying freshly read columns into 16 MiB pages
  post-filter      the predicate applied after the read (pre-filter caching)
O_DIRECT runs: experiment/FA_* (TPC-H 3 runs, ClickBench 2); RAM runs: experiment/RAM2_*.
Prefetch off. TPC-H SF50 4 GB worst order; ClickBench 6 GB natural order.

  pixi run -e duckdb-python python result/plot_scan_path_breakdown.py [--ram]
"""
from __future__ import annotations

import collections
import glob
import json
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

_src = (OUT / "tpch_sf50_per_query.py").read_text().split("FIELDS = ")[0]
_ns: dict = {"__file__": str(OUT / "tpch_sf50_per_query.py")}
exec(compile(_src, "tpch_sf50_per_query.py", "exec"), _ns)
P = _ns["P"]

# "paging + reorder" is the reordered batch together with order-aware eviction (the cache is told
# the batch's remaining query order and evicts the column needed furthest ahead first).
CONDS = [("baseline", "base"), ("paging", "lru"), ("paging\n+ reorder", "ro_evict")]
WORKLOADS = [("TPC-H SF50", "th", (1, 2, 3)), ("ClickBench", "cb", (1, 2))]
PARTS = [("reserve wait", "#e8a33d"), ("disk wait", "#b4522e"),
         ("decode", "#3d6b66"), ("cache serve", "#9fb8b4"), ("cache populate", "#6a3d9a"),
         ("post-filter", "#c9b3e0")]


def log_sums(run: str) -> dict[str, float]:
    """Per-execution sums of the scan-path timers in the engine log."""
    log = glob.glob(str(EXP / run / "*" / "log_dir" / "*.log"))[0]
    out = collections.Counter()
    execution = None
    g = lambda line, k: int(m[1]) / 1e6 if (m := re.search(rf" {k}=(\d+)", line)) else 0.0  # noqa: E731
    for line in open(log):
        m = re.search(r"sirius_set_query_label\('\w+_q\d+_exec(\d+)'\)", line)
        if m:
            execution = int(m[1])
            continue
        if execution is None or execution < 2:
            continue
        if "[scan-audit] parquet_materialize" in line:
            out["read"] += g(line, "read_us") + g(line, "ps_read_us")
            out["cache serve"] += (g(line, "lookup_us") + g(line, "materialize_us") +
                                   g(line, "assemble_us") + g(line, "ps_peek_us") +
                                   g(line, "ps_assemble_us"))
        elif "[scan-populate] store=pages us=" in line:
            out["cache populate"] += g(line, "us")
        elif "[scan-stage]" in line:
            out["post-filter"] += g(line, "postfilter_us")
    return {k: v / 2 for k, v in out.items()}


def task_waits(run: str) -> dict[str, float]:
    """Per-execution Reserving / Queued time of scan tasks, and wall time outside all tasks."""
    tel = next((EXP / run).glob("*/telemetry_data"))
    qlabel, plan_q, pipe_plan, chain = P.load_telemetry(tel)
    events = collections.defaultdict(list)
    pipe = {}
    for path in sorted(tel.rglob("task/*.ndjson")):
        for line in path.open():
            o = json.loads(line)
            st = o.get("data", {}).get("state")
            events[o["id"]].append((o["timestamp"], st))
            if isinstance(st, dict) and isinstance(st.get("Created"), dict):
                pipe[o["id"]] = st["Created"].get("pipeline_uuid", "")
    out = collections.Counter()
    for tid, ser in events.items():
        pid = pipe.get(tid)
        label = P.resolve_query_label(pid, plan_q, pipe_plan, qlabel) if pid else None
        parsed = P.parse_label(label) if label else None
        if not parsed or parsed[2] < 2 or "GPU_SCAN" not in chain.get(pid, {}).values():
            continue
        ser.sort(key=lambda e: e[0])
        for (ts, st), (te, _) in zip(ser, ser[1:]):
            name = next(iter(st)) if isinstance(st, dict) and st else str(st)
            if name == "Reserving":
                out["reserve wait"] += (te - ts) / 1e9
            elif name == "Queued":
                out["queued"] += (te - ts) / 1e9
    return {k: v / 2 for k, v in out.items()}


def outside(run: str) -> float:
    spans = _ns["task_spans"](run)
    w = _ns["walls"](run)
    gaps = []
    for (q, e), s in spans.items():
        if s and (q, e) in w:
            gaps.append(w[(q, e)] - (max(b for _, b, _ in s) - min(a for a, _, _ in s)) / 1e9)
    return sum(gaps) / 2


def main() -> int:
    # --ram: the RAM2_* runs alone (bytes in host RAM, so no disk wait); "decode" is then their
    # whole read time -- page-cache read, host->GPU copy, decompress, decode.
    ram_mode = "--ram" in sys.argv
    parts = [p for p in PARTS if not (ram_mode and p[0] == "disk wait")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ax, (title, bench, reps) in zip(axes, WORKLOADS):
        top = 0.0
        base_total = None
        for x, (label, cond) in enumerate(CONDS):
            od_runs = ([f"RAM2_{bench}_{cond}"] if ram_mode
                       else [f"FA_{bench}_{cond}_r{r}" for r in reps])
            od = [log_sums(r) for r in od_runs]
            mean = lambda k: statistics.mean(d.get(k, 0.0) for d in od)  # noqa: E731
            ram = log_sums(f"RAM2_{bench}_{cond}")
            waits = [task_waits(r) for r in od_runs]
            segs = {
                "reserve wait": statistics.mean(w.get("reserve wait", 0.0) for w in waits),
                "queued": statistics.mean(w.get("queued", 0.0) for w in waits),
                "disk wait": 0.0 if ram_mode else max(mean("read") - ram.get("read", 0.0), 0.0),
                "decode": ram.get("read", 0.0),
                "cache serve": mean("cache serve"),
                "cache populate": mean("cache populate"),
                "post-filter": mean("post-filter"),
            }
            setup = statistics.mean(outside(r) for r in od_runs)
            bottom = 0.0
            for key, color in parts:
                ax.bar(x, segs[key], 0.6, bottom=bottom, color=color, edgecolor="#333333",
                       linewidth=0.5, zorder=3, label=key if x == 0 else None)
                bottom += segs[key]
            if base_total is None:
                base_total = bottom
                caption = f"{bottom:.0f}s"
            else:
                caption = f"{bottom:.0f}s\n{100 * (bottom / base_total - 1):+.1f}%"
            ax.text(x, bottom + 1.0, caption, ha="center", va="bottom", fontsize=6.5)
            top = max(top, bottom)
            print(f"{title:11} {label.replace(chr(10), ' '):32} setup(wall)={setup:.2f}s  "
                  + "  ".join(f"{k}={v:.2f}" for k, v in segs.items()) + f"  total={bottom:.1f}")
        ax.set_xticks(range(len(CONDS)))
        ax.set_xticklabels([c[0] for c in CONDS], fontsize=6.3)
        ax.set_ylim(0, top * 1.25)
        ps.finish(ax, "scan path, summed over scan tasks (s)" if ax is axes[0] else None)
        ax.set_title(title, fontsize=8.5, pad=4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.5, frameon=False, ncol=7, loc="lower center",
               bbox_to_anchor=(0.5, 0.04))
    fig.text(0.01, 0.005, "paging + reorder: reordered batch with order-aware eviction "
             "(the cache evicts the column the batch needs furthest ahead first).",
             fontsize=6.5, color="#555555")
    fig.tight_layout(rect=(0, 0.1, 1, 0.94 if ram_mode else 1))
    stem = "scan_path_breakdown_ram" if ram_mode else "scan_path_breakdown"
    if ram_mode:
        fig.suptitle("parquet bytes in host RAM (no disk wait), 1 run per condition",
                     fontsize=7.5, x=0.01, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300)
    print(OUT / f"{stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
