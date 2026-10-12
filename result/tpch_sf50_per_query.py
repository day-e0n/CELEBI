#!/usr/bin/env python3
"""TPC-H SF50, 4GB page cache, worst arrival order, prefetch off: per-query performance.

Three runs per condition (experiment/RO_worst_{base,cache,reorder}{,_r2,_r3}); every number is
the mean over those runs and executions 2-3 (execution 1 is warmup).

Outputs next to this file:
  tpch_sf50_per_query.png        per-query wall, baseline / cache / cache + reorder
  tpch_sf50_breakdown.png        each query's wall split into exclusive parts (see PARTS)
  tpch_sf50_per_query.csv        all numbers, plus summed scan internals from the engine log

The breakdown parts add up to wall time. "scan, reservation wait" is time when a scan is
running, a worker slot is free and a task is stuck in Reserving -- the other scans keep going
during it, so it is NOT all lost. `resv_lost_s` estimates what is lost: the free slots that a
reserving task could have filled, as a share of the 4 slots, integrated over time.

  pixi run -e duckdb-python python result/tpch_sf50_per_query.py
"""
import collections
import csv
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
SLOTS = 4
EXECS = {2, 3}
CONDS = [("base", "Baseline (no cache)", "#b4b2ab"),
         ("cache", "Page cache", "#86b6ef"),
         ("reorder", "Page cache + reorder", "#2a78d6")]
RUNS = {c: [f"RO_worst_{c}", f"RO_worst_{c}_r2", f"RO_worst_{c}_r3"] for c, *_ in CONDS}
PARTS = [("scan_full", "scan, slots full", "#2a78d6"),
         ("scan_resv", "scan, reservation wait", "#eb6834"),
         ("scan_idle", "scan, slots idle", "#1baf7a"),
         ("other_ops", "other operators", "#eda100"),
         ("idle", "idle / outside tasks", "#b4b2ab")]
INNER = ["disk_read", "cache_prep", "populate", "postfilter"]
QUERIES = list(range(1, 23))

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
    """Sweep one query's spans into the exclusive parts plus the lost-slot estimate (seconds)."""
    out = collections.Counter()
    events = sorted([(s, 1, k) for s, e, k in spans] + [(e, -1, k) for s, e, k in spans])
    live = collections.Counter()
    prev = None
    for t, d, k in events:
        if prev is not None and t > prev:
            dur = (t - prev) / 1e9
            busy = live["scan"] + live["compute"]
            if live["reserve"] > 0 and busy < SLOTS:
                out["resv_lost_s"] += min(SLOTS - busy, live["reserve"]) / SLOTS * dur
            if live["scan"] > 0:
                if busy >= SLOTS:
                    out["scan_full"] += dur
                elif live["reserve"] > 0:
                    out["scan_resv"] += dur
                else:
                    out["scan_idle"] += dur
            elif live["compute"] > 0:
                out["other_ops"] += dur
            out["scan_task_s"] += live["scan"] * dur
        live[k] += d
        prev = t
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


FIELDS = (["wall"] + [p for p, *_ in PARTS] + ["resv_lost_s", "scan_task_s"] + INNER)
data = {}  # (cond, q) -> Counter of means
for cond, *_ in CONDS:
    acc = collections.defaultdict(collections.Counter)
    n = 0
    for run in RUNS[cond]:
        sp, w, inner = task_spans(run), walls(run), scan_internals(run)
        for q in QUERIES:
            for e in EXECS:
                parts = split(sp.get((q, e), []))
                parts["wall"] = w[(q, e)]
                parts["idle"] = parts["wall"] - sum(parts[k] for k, *_ in PARTS[:4])
                parts.update(inner.get((q, e), {}))
                acc[q].update(parts)
        n += len(EXECS)
    for q in QUERIES:
        data[(cond, q)] = collections.Counter({k: v / n for k, v in acc[q].items()})

with open(OUT / "tpch_sf50_per_query.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "query"] + FIELDS)
    for cond, *_ in CONDS:
        for q in QUERIES:
            w.writerow([cond, f"q{q}"] + [f"{data[(cond, q)][k]:.3f}" for k in FIELDS])
        w.writerow([cond, "total"] + [f"{sum(data[(cond, q)][k] for q in QUERIES):.3f}" for k in FIELDS])

BG = "#fcfcfb"


def style(ax):
    ax.set_facecolor(BG)
    ax.set_xticks(range(22), [f"q{q}" for q in QUERIES], fontsize=9, color="#52514e")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0, colors="#898781", labelsize=9)
    ax.set_ylabel("Wall time (s)", fontsize=9, color="#52514e")
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#898781")
    ax.set_xlim(-0.6, 21.6)


tot = {c: sum(data[(c, q)]["wall"] for q in QUERIES) for c, *_ in CONDS}
subtitle = (f"Total: baseline {tot['base']:.1f}s, cache {tot['cache']:.1f}s "
            f"({(tot['cache'] / tot['base'] - 1) * 100:+.1f}%), cache + reorder {tot['reorder']:.1f}s "
            f"({(tot['reorder'] / tot['base'] - 1) * 100:+.1f}%)")
note = ("Lower is better. Worst arrival order, prefetch OFF, 3 runs x executions 2-3 averaged.")

# Per-query wall.
fig, ax = plt.subplots(figsize=(12, 4.2), dpi=170)
fig.patch.set_facecolor(BG)
bw = 0.27
for i, (cond, label, color) in enumerate(CONDS):
    ax.bar([q - 1 + (i - 1) * bw for q in QUERIES], [data[(cond, q)]["wall"] for q in QUERIES],
           bw, color=color, edgecolor=BG, linewidth=1, zorder=2, label=label)
style(ax)
ax.set_title(subtitle, fontsize=10, color="#0b0b0b", loc="left")
ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
fig.suptitle("TPC-H SF50, 4 GB GPU page cache: per-query wall time", fontsize=11, x=0.01, ha="left")
fig.text(0.01, 0.01, note, fontsize=7.5, color="#898781")
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
fig.savefig(OUT / "tpch_sf50_per_query.png", facecolor=BG)
plt.close(fig)

# Breakdown.
fig, ax = plt.subplots(figsize=(15, 4.8), dpi=160)
fig.patch.set_facecolor(BG)
for ci, (cond, *_) in enumerate(CONDS):
    xs = [q - 1 + (ci - 1) * (bw + 0.02) for q in QUERIES]
    bottom = [0.0] * 22
    for key, label, color in PARTS:
        vals = [max(data[(cond, q)][key], 0.0) for q in QUERIES]
        ax.bar(xs, vals, bw, bottom=bottom, color=color, edgecolor=BG, linewidth=0.6,
               label=label if ci == 0 else None, zorder=2)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for x, top in zip(xs, bottom):
        ax.text(x, top + 0.04, "BCR"[ci], ha="center", va="bottom", fontsize=6.5, color="#52514e")
style(ax)
ax.set_title(subtitle, fontsize=10, color="#0b0b0b", loc="left")
ax.legend(frameon=False, fontsize=8.5, ncol=5, loc="upper center")
fig.suptitle("TPC-H SF50, 4 GB GPU page cache: per-query wall time breakdown "
             "(B = baseline, C = cache, R = cache + reorder)", fontsize=11, x=0.01, ha="left")
fig.text(0.01, 0.01, note + " Parts are exclusive and add up to wall time.",
         fontsize=7.5, color="#898781")
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
fig.savefig(OUT / "tpch_sf50_breakdown.png", facecolor=BG)
plt.close(fig)

for cond, *_ in CONDS:
    print(cond, " ".join(f"{k}={sum(data[(cond, q)][k] for q in QUERIES):.2f}" for k in FIELDS))
