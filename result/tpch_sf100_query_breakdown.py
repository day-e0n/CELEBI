#!/usr/bin/env python3
"""TPC-H SF100, 6GB page cache, prefetch off: per-query wall time broken into exclusive parts.

For every query (executions 2-3 averaged) the wall clock is split, instant by instant, by what
the query's tasks were doing (task telemetry):

  scan, slots full        >=1 GPU_SCAN computing and all 4 worker slots busy
  scan, reservation wait  >=1 GPU_SCAN computing, a slot free, and a task stuck in Reserving
  scan, slots idle        >=1 GPU_SCAN computing, a slot free, and no task waiting for it
  other operators         no scan, some other operator computing
  idle / outside tasks    nothing computing (incl. planning and result fetch outside tasks)

These five add up to the query's wall time. The CSV also carries summed (not exclusive) scan
internals from the engine log: disk read, cache prep (probe + partial-serve reads), page
populate, and post-filter, each summed over the query's scan batches.

  pixi run -e duckdb-python python result/tpch_sf100_query_breakdown.py
Reads experiment/H_{worst,natural}_{base,cache,reorder}/ ; writes tpch_sf100_query_breakdown.{csv,png}.
"""
import collections
import csv
import datetime as dt
import importlib.util
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiment"
OUT = Path(__file__).resolve().parent
ORDERS = ["worst", "natural"]
CONDS = [("base", "Baseline"), ("cache", "Cache"), ("reorder", "Cache+reorder")]
EXECS = {2, 3}
PARTS = [("scan_full", "scan, slots full", "#2a78d6"),
         ("scan_resv", "scan, reservation wait", "#eb6834"),
         ("scan_idle", "scan, slots idle", "#1baf7a"),
         ("other_ops", "other operators", "#eda100"),
         ("idle", "idle / outside tasks", "#b4b2ab")]

spec = importlib.util.spec_from_file_location("P", ROOT / "scripts/parse_quent_operator_breakdown.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def task_spans(run):
    """(query, exec) -> [(start_ns, end_ns, kind)], kind in scan/compute/reserve/waiting."""
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
    spans = collections.defaultdict(list)
    for tid, ser in events.items():
        pid = pipe.get(tid)
        label = P.resolve_query_label(pid, plan_q, pipe_plan, qlabel) if pid else None
        parsed = P.parse_label(label) if label else None
        if not parsed or parsed[2] not in EXECS:
            continue
        ser.sort(key=lambda e: e[0])
        ops = chain.get(pid, {})
        for (ts, st), (te, _) in zip(ser, ser[1:]):
            name = next(iter(st)) if isinstance(st, dict) and st else str(st)
            if name == "Computing":
                op = st["Computing"].get("current_operator_id") if isinstance(st["Computing"], dict) else None
                kind = "scan" if ops.get(op) == "GPU_SCAN" else "compute"
            elif name == "Reserving":
                kind = "reserve"
            elif name in ("Finished", "Failed", "Cancelled"):
                continue
            else:
                kind = "waiting"
            spans[(int(parsed[1]), parsed[2])].append((ts, te, kind))
    return spans


def split(spans):
    """Sweep one query's spans into the exclusive parts (seconds)."""
    out = collections.Counter()
    events = sorted([(s, 1, k) for s, e, k in spans] + [(e, -1, k) for s, e, k in spans])
    live = collections.Counter()
    prev = None
    for t, d, k in events:
        if prev is not None and t > prev:
            dur = (t - prev) / 1e9
            if live["scan"] > 0:
                if live["scan"] + live["compute"] >= 4:
                    out["scan_full"] += dur
                elif live["reserve"] > 0:
                    out["scan_resv"] += dur
                else:
                    out["scan_idle"] += dur
            elif live["compute"] > 0:
                out["other_ops"] += dur
        live[k] += d
        prev = t
    out["span"] = (events[-1][0] - events[0][0]) / 1e9 if events else 0.0
    return out


def walls(run):
    w = {}
    for line in open(EXP / run / "runner.log"):
        m = re.search(r"exec=(\d+)/\d+ .* q(\d+) done wall=([\d.]+)s", line)
        if m and int(m[1]) in EXECS:
            w[(int(m[2]), int(m[1]))] = float(m[3])
    return w


def scan_internals(run):
    """(query, exec) -> summed seconds of disk read / cache prep / populate / post-filter."""
    log = next((EXP / run).glob("*/log_dir/*.log"))
    out = collections.defaultdict(collections.Counter)
    key = None
    for line in open(log):
        m = re.search(r"sirius_set_query_label\('\w+_q(\d+)_exec(\d+)'\)", line)
        if m:
            key = (int(m[1]), int(m[2]))
            continue
        if not key or key[1] not in EXECS:
            continue
        if "[scan-audit] parquet_materialize" in line:
            out[key]["disk_read"] += int(re.search(r" read_us=(\d+)", line)[1]) / 1e6
            out[key]["cache_prep"] += int(re.search(r"stage_prep_us=(\d+)", line)[1]) / 1e6
        elif "[scan-populate]" in line and (m := re.search(r"store=\w+ us=(\d+)", line)):
            out[key]["populate"] += int(m[1]) / 1e6
        elif "[scan-stage]" in line:
            out[key]["postfilter"] += int(re.search(r"postfilter_us=(\d+)", line)[1]) / 1e6
    return out


rows = []
for order in ORDERS:
    for cond, _ in CONDS:
        run = f"H_{order}_{cond}"
        sp, w, inner = task_spans(run), walls(run), scan_internals(run)
        for q in range(1, 23):
            acc = collections.Counter()
            for e in EXECS:
                parts = split(sp.get((q, e), []))
                wall = w[(q, e)]
                parts["idle"] = wall - sum(parts[k] for k, *_ in PARTS[:4])
                parts["wall"] = wall
                parts.update(inner.get((q, e), {}))
                for k, v in parts.items():
                    acc[k] += v / len(EXECS)
            rows.append({"order": order, "condition": cond, "query": f"q{q}", **{
                k: round(acc[k], 3) for k in ["wall"] + [p for p, *_ in PARTS] +
                ["disk_read", "cache_prep", "populate", "postfilter"]}})

with open(OUT / "tpch_sf100_query_breakdown.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

bg = "#fcfcfb"
fig, axes = plt.subplots(2, 1, figsize=(15, 8), dpi=160, sharey=True)
fig.patch.set_facecolor(bg)
bw = 0.27
for ax, order in zip(axes, ORDERS):
    ax.set_facecolor(bg)
    for ci, (cond, _) in enumerate(CONDS):
        sub = {r["query"]: r for r in rows if r["order"] == order and r["condition"] == cond}
        xs = [q - 1 + (ci - 1) * (bw + 0.02) for q in range(1, 23)]
        bottom = [0.0] * 22
        for key, label, color in PARTS:
            vals = [max(sub[f"q{q}"][key], 0.0) for q in range(1, 23)]
            ax.bar(xs, vals, bw, bottom=bottom, color=color, edgecolor=bg, linewidth=0.6,
                   label=label if ci == 0 else None, zorder=2)
            bottom = [b + v for b, v in zip(bottom, vals)]
        for x, top in zip(xs, bottom):
            ax.text(x, top + 0.05, "BCR"[ci], ha="center", va="bottom", fontsize=6.5, color="#52514e")
    ax.set_xticks(range(22), [f"q{q}" for q in range(1, 23)], fontsize=9, color="#52514e")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0, colors="#898781", labelsize=9)
    ax.set_ylabel("Wall time (s)", fontsize=9, color="#52514e")
    ax.set_title(f"{order} arrival order", fontsize=10, color="#0b0b0b", loc="left")
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#898781")
    ax.set_xlim(-0.6, 21.6)
axes[0].legend(frameon=False, fontsize=8.5, ncol=5, loc="upper center")
fig.suptitle("TPC-H SF100, 6 GB GPU page cache: per-query wall time breakdown "
             "(B = baseline, C = cache, R = cache + reorder)", fontsize=11, x=0.01, ha="left")
fig.text(0.01, 0.005, "Prefetch OFF, executions 2-3 averaged. Parts are exclusive and add up to wall "
         "time. Full numbers incl. disk read / populate: tpch_sf100_query_breakdown.csv",
         fontsize=7.5, color="#898781")
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
fig.savefig(OUT / "tpch_sf100_query_breakdown.png", facecolor=bg)
print(OUT / "tpch_sf100_query_breakdown.png")
