#!/usr/bin/env python3
"""Per query: did the page cache help, do nothing, or hurt -- and why.

Compares baseline with paging + reorder (reordered batch + order-aware eviction), per query,
averaged over runs and executions 2-3 (O_DIRECT, prefetch off; experiment/FA_*):
  wall_delta     end-to-end wall change (s)
  read_saved     scan read time saved, summed over scan tasks (s): disk + decode not done
  populate       time spent copying freshly read columns into pages, summed over scan tasks
  full_hits      scan batches served entirely from the cache
  reserve_delta  change in scan tasks' Reserving time, summed (s)
Category: helped (wall -5% or more), hurt (wall +5% or more), flat otherwise.

  pixi run -e duckdb-python python result/per_query_cache_effect.py
writes per_query_cache_effect_{tpch,clickbench}.csv and per_query_cache_effect.png
"""
from __future__ import annotations

import collections
import csv
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

WORKLOADS = [("tpch", "TPC-H SF50", "th", (1, 2, 3)), ("clickbench", "ClickBench", "cb", (1, 2))]
CONDS = {"base": "base", "best": "ro_evict"}


def per_query(run: str) -> dict[int, collections.Counter]:
    out: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for line in open(EXP / run / "runner.log"):
        m = re.search(r"exec=(\d+)/\d+ .* q(\d+) done wall=([\d.]+)s", line)
        if m and int(m[1]) >= 2:
            out[int(m[2])]["wall"] += float(m[3]) / 2
    log = glob.glob(str(EXP / run / "*" / "log_dir" / "*.log"))[0]
    q = ex = None
    g = lambda line, k: int(m[1]) / 1e6 if (m := re.search(rf" {k}=(\d+)", line)) else 0.0  # noqa: E731
    for line in open(log):
        m = re.search(r"sirius_set_query_label\('\w+_q(\d+)_exec(\d+)'\)", line)
        if m:
            q, ex = int(m[1]), int(m[2])
            continue
        if ex is None or ex < 2:
            continue
        if "[scan-audit] parquet_materialize" in line:
            out[q]["read"] += (g(line, "read_us") + g(line, "ps_read_us")) / 2
            out[q]["full_hits"] += (1 if " full_hit=1" in line else 0) / 2
            out[q]["batches"] += 0.5
        elif "[scan-populate] store=pages us=" in line:
            out[q]["populate"] += g(line, "us") / 2
    # Reserving time of scan tasks, per query.
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
    for tid, ser in events.items():
        pid = pipe.get(tid)
        label = P.resolve_query_label(pid, plan_q, pipe_plan, qlabel) if pid else None
        parsed = P.parse_label(label) if label else None
        if not parsed or parsed[2] < 2 or "GPU_SCAN" not in chain.get(pid, {}).values():
            continue
        ser.sort(key=lambda e: e[0])
        for (ts, st), (te, _) in zip(ser, ser[1:]):
            if (next(iter(st)) if isinstance(st, dict) and st else str(st)) == "Reserving":
                out[int(parsed[1])]["reserve"] += (te - ts) / 1e9 / 2
    return out


def mean_runs(bench: str, cond: str, reps) -> dict[int, dict[str, float]]:
    runs = [per_query(f"FA_{bench}_{cond}_r{r}") for r in reps]
    queries = set.intersection(*(set(r) for r in runs))
    return {q: {k: statistics.mean(r[q][k] for r in runs) for k in
                ("wall", "read", "populate", "full_hits", "batches", "reserve")} for q in queries}


def main() -> int:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8))
    for ax, (slug, title, bench, reps) in zip(axes, WORKLOADS):
        base, best = mean_runs(bench, "base", reps), mean_runs(bench, "ro_evict", reps)
        rows = []
        for q in sorted(base):
            b, c = base[q], best[q]
            delta = c["wall"] - b["wall"]
            pct = 100 * delta / b["wall"]
            cat = "helped" if pct <= -5 else "hurt" if pct >= 5 else "flat"
            rows.append({"query": f"q{q}", "base_wall": round(b["wall"], 3),
                         "best_wall": round(c["wall"], 3), "wall_delta": round(delta, 3),
                         "wall_pct": round(pct, 1), "category": cat,
                         "read_base": round(b["read"], 2), "read_best": round(c["read"], 2),
                         "read_saved": round(b["read"] - c["read"], 2),
                         "populate": round(c["populate"], 2),
                         "full_hits": round(c["full_hits"], 1), "batches": round(c["batches"], 1),
                         "reserve_delta": round(c["reserve"] - b["reserve"], 2)})
        with open(OUT / f"per_query_cache_effect_{slug}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        colors = {"helped": "#3d6b66", "flat": "#b4b2ab", "hurt": "#b4522e"}
        xs = range(len(rows))
        ax.bar(list(xs), [r["wall_pct"] for r in rows], 0.7, zorder=3,
               color=[colors[r["category"]] for r in rows], edgecolor="#333333", linewidth=0.4)
        ax.axhline(0, color="#333333", linewidth=0.6)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([r["query"] for r in rows], fontsize=5.5, rotation=90)
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ps.finish(ax, "wall change vs baseline (%)")
        ax.set_title(f"{title}: paging + reorder (order-aware eviction) vs baseline, per query",
                     fontsize=8, pad=4)
        n = collections.Counter(r["category"] for r in rows)
        print(f"{title}: helped {n['helped']}  flat {n['flat']}  hurt {n['hurt']}")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"per_query_cache_effect.{ext}", dpi=300)
    print(OUT / "per_query_cache_effect.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
