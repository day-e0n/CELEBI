#!/usr/bin/env python3
"""TPC-H SF50: how much disk reading the page cache removes, what it costs, and what reaches wall.

Same runs as the headline figure (4 GB, worst arrival order, prefetch off, O_DIRECT,
experiment/RO_worst_{base,cache,reorder}{,_r2,_r3}); every number is the mean over the 3 runs
of executions 2-3.

Left: scan work summed over all scan tasks (the 4 worker slots run up to 4 at once, so these
add up to more than wall time):
  disk read      read_us + stage_prep_us of every scan batch. stage_prep_us is the partial-serve
                 (B4) read of the rows a part-cached batch lacks; measured at 98% disk read
                 (experiment/PS_nosync_*).
  cache populate copying freshly read columns into 16 MiB pages ([scan-populate]).
Right: end-to-end wall time of the 22 queries.

  pixi run -e duckdb-python python result/plot_disk_read_tpch.py
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

EXP = ROOT / "experiment"
OUT = Path(__file__).resolve().parent
CONDS = [("baseline", "base", ps.NEUTRAL), ("paging", "cache", ps.DARK),
         ("paging\n+ reorder", "reorder", ps.ACCENT)]
REPS = ("", "_r2", "_r3")


def run_numbers(run: str) -> dict[str, float]:
    """Per-execution means (executions 2-3) of disk read, populate and wall for one run."""
    log = glob.glob(str(EXP / run / "*" / "log_dir" / "*.log"))[0]
    disk = populate = 0.0
    execution = None
    for line in open(log):
        m = re.search(r"sirius_set_query_label\('\w+_q\d+_exec(\d+)'\)", line)
        if m:
            execution = int(m[1])
            continue
        if execution is None or execution < 2:
            continue
        if "[scan-audit] parquet_materialize" in line:
            disk += int(re.search(r" read_us=(\d+)", line)[1]) / 1e6
            disk += int(re.search(r"stage_prep_us=(\d+)", line)[1]) / 1e6
        elif "[scan-populate]" in line and (p := re.search(r"store=\w+ us=(\d+)", line)):
            populate += int(p[1]) / 1e6
    wall = {}
    for line in open(EXP / run / "runner.log"):
        m = re.search(r"exec=(\d+)/\d+ .* done wall=([\d.]+)s", line)
        if m and int(m[1]) >= 2:
            wall[int(m[1])] = wall.get(int(m[1]), 0.0) + float(m[2])
    n = len(wall)
    return {"disk": disk / n, "populate": populate / n, "wall": sum(wall.values()) / n}


def main() -> int:
    data = {}
    for _, key, _ in CONDS:
        runs = [run_numbers(f"RO_worst_{key}{r}") for r in REPS]
        data[key] = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}
        print(key, {k: round(v, 2) for k, v in data[key].items()})
    base = data["base"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    for x, (label, key, style) in enumerate(CONDS):
        d = data[key]
        ax1.bar(x, d["disk"], 0.6, zorder=3, **style)
        ax1.bar(x, d["populate"], 0.6, bottom=d["disk"], zorder=3, color="white",
                edgecolor=style.get("edgecolor", "#333333"), hatch="////", linewidth=0.6)
        top = d["disk"] + d["populate"]
        caption = f"disk {d['disk']:.1f}s" if key == "base" else \
            f"disk {d['disk']:.1f}s ({100 * (d['disk'] / base['disk'] - 1):+.0f}%)\n" \
            f"+ populate {d['populate']:.1f}s"
        ax1.text(x, top + 1.5, caption, ha="center", va="bottom", fontsize=6)
        ax2.bar(x, d["wall"], 0.6, zorder=3, **style)
        caption = f"{d['wall']:.1f}s" if key == "base" else \
            f"{d['wall']:.1f}s\n{100 * (d['wall'] / base['wall'] - 1):+.1f}%"
        ax2.text(x, d["wall"] + 0.5, caption, ha="center", va="bottom", fontsize=6.5)
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(CONDS)))
        ax.set_xticklabels([c[0] for c in CONDS], fontsize=7)
    ax1.set_ylim(0, max(d["disk"] + d["populate"] for d in data.values()) * 1.32)
    ax2.set_ylim(0, max(d["wall"] for d in data.values()) * 1.30)
    ps.finish(ax1, "scan work, summed over tasks (s)")
    ps.finish(ax2, "query time (s)")
    ax1.set_title("(a) disk read and cache populate", fontsize=8, pad=4)
    ax2.set_title("(b) end-to-end time", fontsize=8, pad=4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"disk_read_tpch.{ext}", dpi=300)
    print(OUT / "disk_read_tpch.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
