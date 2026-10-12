#!/usr/bin/env python3
"""How much of the data the scans needed came out of the page cache.

Every scan batch logs (SIRIUS_CACHE_HIT_TRACE=1) the decoded bytes it handed downstream
(total_bytes: the projected columns of the row groups it covers, before any filter) and the
decoded bytes the parquet reader had to produce for it (read_bytes); the rest came from cached
pages. Batch kinds:
  full     every column served from pages, no reader call
  splice   some columns fully cached, the rest read
  partial  some column part-cached (B4): only the missing rows read
  miss     everything read
byte hit rate  = 1 - sum(read_bytes) / sum(total_bytes)
batch hit rate = full batches / all batches
Executions 2-3, 1 run per condition, O_DIRECT, prefetch off, GPU 1 (experiment/HIT_*).
TPC-H SF50 4 GB worst order; ClickBench 6 GB natural order, 41 queries.

  pixi run -e duckdb-python python result/cache_hit_rate.py [--v3]
writes cache_hit_rate.csv, cache_hit_rate_per_table.csv, cache_hit_rate_per_query.csv,
cache_hit_rate.png/.pdf. --v3: the current engine (scan no-history reservation 3x, late
materialization on; experiment/HIT2_*, paging and paging + reorder only), files *_v3.*
"""
from __future__ import annotations

import collections
import csv
import glob
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
EXP = ROOT / "experiment"
sys.path.insert(0, str(ROOT / "scripts"))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

WORKLOADS = [("TPC-H SF50", "th"), ("ClickBench", "cb")]
CONDS = [("paging", "lru"), ("paging\n+ order-aware", "evict"),
         ("paging + reorder\n(LRU eviction)", "ro_lru"), ("paging\n+ reorder", "ro_evict")]
# The figure shows these two; the csv files keep all four.
PLOT_CONDS = [("paging", "lru"), ("paging + reorder", "ro_evict")]
KINDS = ("full", "splice", "partial", "miss")
GIB = 1024 ** 3
LINE = re.compile(r"\[cache-hit\] file='([^']*)' total_bytes=(\d+) read_bytes=(\d+) kind=(\w+)")


def collect(run: str):
    """Per (query, table): total / read bytes and batch kinds, summed over executions 2-3."""
    log = glob.glob(str(EXP / run / "*" / "log_dir" / "*.log"))[0]
    acc: dict[tuple[int, str], collections.Counter] = collections.defaultdict(collections.Counter)
    execs: set[int] = set()
    q = ex = None
    for line in open(log):
        m = re.search(r"sirius_set_query_label\('\w+_q(\d+)_exec(\d+)'\)", line)
        if m:
            q, ex = int(m[1]), int(m[2])
            continue
        if ex is None or ex < 2:
            continue
        m = LINE.search(line)
        if m:
            execs.add(ex)
            c = acc[(q, Path(m[1]).stem)]
            c["total"] += int(m[2])
            c["read"] += int(m[3])
            c[m[4]] += 1
    n = len(execs) or 1
    return {k: {f: v / n for f, v in c.items()} for k, c in acc.items()}


def summarize(rows) -> dict[str, float]:
    s = collections.Counter()
    for c in rows:
        s.update(c)
    batches = sum(s[k] for k in KINDS)
    return {"total_gib": s["total"] / GIB, "read_gib": s["read"] / GIB,
            "served_gib": (s["total"] - s["read"]) / GIB,
            "byte_hit_pct": 100 * (1 - s["read"] / s["total"]) if s["total"] else 0.0,
            "batches": batches,
            **{f"{k}_batches": s[k] for k in KINDS},
            "batch_hit_pct": 100 * s["full"] / batches if batches else 0.0}


def main() -> int:
    global CONDS, PLOT_CONDS
    prefix, suffix = "HIT", ""
    if "--v3" in sys.argv:
        prefix, suffix = "HIT2", "_v3"
        CONDS = [("paging", "lru"), ("paging\n+ reorder", "best")]
        PLOT_CONDS = [("paging", "lru"), ("paging + reorder", "best")]
    overall, per_table, per_query = [], [], []
    for title, bench in WORKLOADS:
        base = collect(f"{prefix}_{bench}_base")
        if base:
            b = summarize(base.values())
            print(f"{title:11} baseline: scans need {b['total_gib']:.1f} GiB per execution "
                  f"(after reader filter), {b['batches']:.0f} batches")
        for label, cond in CONDS:
            data = collect(f"{prefix}_{bench}_{cond}")
            s = summarize(data.values())
            name = label.replace("\n", " ")
            overall.append({"workload": title, "condition": name, **{k: round(v, 2) for k, v in s.items()}})
            for t in sorted({t for _, t in data}):
                st = summarize(c for (_, tt), c in data.items() if tt == t)
                per_table.append({"workload": title, "condition": name, "table": t,
                                  **{k: round(v, 2) for k, v in st.items()}})
            for qq in sorted({qq for qq, _ in data}):
                sq = summarize(c for (q2, _), c in data.items() if q2 == qq)
                per_query.append({"workload": title, "condition": name, "query": f"q{qq}",
                                  **{k: round(v, 2) for k, v in sq.items()}})
            print(f"{title:11} {name:30} need {s['total_gib']:6.1f} GiB  served {s['served_gib']:6.1f}  "
                  f"read {s['read_gib']:6.1f}  byte hit {s['byte_hit_pct']:5.1f}%  batches {s['batches']:.0f} "
                  + " ".join(f"{k}={s[f'{k}_batches']:.0f}" for k in KINDS)
                  + f"  batch hit {s['batch_hit_pct']:.1f}%")

    # Byte hit rate per condition, one line per workload.
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    styles = {"TPC-H SF50": dict(color="#7a3a22", marker="o"),
              "ClickBench": dict(color="#3d6b66", marker="s")}
    xs = list(range(len(PLOT_CONDS)))
    for title, _ in WORKLOADS:
        by_cond = {r["condition"]: r["byte_hit_pct"] for r in overall if r["workload"] == title}
        ys = [by_cond[label] for label, _ in PLOT_CONDS]
        ax.plot(xs, ys, linewidth=1.4, markersize=5, zorder=3, label=title, **styles[title])
        below = title == "TPC-H SF50"  # TPC-H labels under the points, ClickBench above
        for x, y in zip(xs, ys):
            ax.text(x, y - 1.2 if below else y + 1.0, f"{y:.1f}%", ha="center",
                    va="top" if below else "bottom", fontsize=6.5, color=styles[title]["color"])
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in PLOT_CONDS], fontsize=7)
    ax.set_xlim(-0.4, len(PLOT_CONDS) - 0.6)
    ax.set_ylim(0, max(32, 1.2 * max(r["byte_hit_pct"] for r in overall)))
    ps.finish(ax, "cache hit ratio (% of bytes)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"cache_hit_rate{suffix}.{ext}", dpi=300)
    for name, rows in (("cache_hit_rate", overall), ("cache_hit_rate_per_table", per_table),
                       ("cache_hit_rate_per_query", per_query)):
        with open(OUT / f"{name}{suffix}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(OUT / f"cache_hit_rate{suffix}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
