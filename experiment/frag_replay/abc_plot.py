"""A -> B -> C: the 6 GiB cache budget and where each query's data came from, batch vs page.

A = q16 (UserID), B = q17 (UserID, SearchPhrase), C = q19 (UserID, EventTime, SearchPhrase),
batches as recorded by a fresh Sirius run (experiment/ABC_base); allocations measured on a
6 GiB-capped cudaMallocAsync pool (replay.cu over abc_trace.py's traces).

  pixi run -e duckdb-python python experiment/frag_replay/abc_plot.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
GIB = 2 ** 30
TEXT, MUTED, SURFACE = "#16161a", "#5a5a60", "#fbfbf9"
BLUE, YELLOW, VIOLET, FREE, DISK = "#2a78d6", "#eda100", "#4a3aa7", "#e4e2dc", "#9a9893"
DESIGNS = [("batch", "Batch"), ("page", "16 MiB page")]
QUERIES = [("A", "A: UserID"), ("B", "B: + SearchPhrase"), ("C", "C: + EventTime")]

content = json.load(open(HERE / "abc_content.json"))
meas = {p: [{k: int(v) for k, v in r.items()} for r in csv.DictReader(open(HERE / f"abc_{p}.csv"))]
        for p, _ in DESIGNS}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbcb6")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_facecolor(SURFACE)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE,
                               gridspec_kw={"width_ratios": [1.25, 1]})
style(ax1)
style(ax2)
width, gap = 0.36, 0.04
for qi, (q, _) in enumerate(QUERIES):
    for di, (p, name) in enumerate(DESIGNS):
        m, c = meas[p][qi], content[p][qi]
        free_total = m["budget"] - m["pool_used"]
        segs = [(c["unique"] / GIB, BLUE, None, "white"),
                (c["duplicate"] / GIB, BLUE, "///", "white"),
                ((m["live_requested"] - m["live_payload"]) / GIB, YELLOW, None, TEXT),
                (max(0, free_total - m["largest_free"]) / GIB, VIOLET, None, "white"),
                (min(free_total, m["largest_free"]) / GIB, FREE, None, TEXT)]
        x = qi + (di - 0.5) * (width + gap)
        bottom = 0.0
        for h, color, hatch, ink in segs:
            if h <= 0:
                continue
            ax1.bar(x, h, width, bottom=bottom, color=color, hatch=hatch, edgecolor=SURFACE,
                    linewidth=2)
            if h >= 0.4:
                box = dict(boxstyle="round,pad=0.2", fc=BLUE, ec="none") if hatch else None
                ax1.text(x, bottom + h / 2, f"{h:.2f}", ha="center", va="center", fontsize=8.5,
                         color=ink, bbox=box)
            bottom += h
        ax1.text(x, -0.18, name, ha="center", va="top", fontsize=8.5, color=MUTED)
ax1.set_xticks(range(3), [f"\n\nafter {lbl}" for _, lbl in QUERIES], fontsize=9, color=TEXT)
ax1.tick_params(axis="x", length=0)
ax1.set_ylim(0, 6.0)
ax1.set_ylabel("GiB of the 6 GiB cache budget", color=MUTED, fontsize=9)
ax1.set_title("Cache budget after each query", loc="left", fontsize=11.5, color=TEXT, pad=46)
ax1.legend([Patch(color=BLUE), Patch(facecolor=BLUE, hatch="///", edgecolor=SURFACE),
            Patch(color=YELLOW), Patch(color=VIOLET), Patch(color=FREE)],
           ["Cached data", "Duplicate copy of cached data", "Internal fragmentation (padding)",
            "External fragmentation (none: nothing evicted)", "Free, usable"],
           loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8,
           labelcolor=TEXT, handlelength=1.4)

for qi, (q, _) in enumerate(QUERIES):
    for di, (p, name) in enumerate(DESIGNS):
        served_cum = [r["served_bytes"] for r in meas[p]]
        served = (served_cum[qi] - (served_cum[qi - 1] if qi else 0)) / GIB
        requested = content[p][qi]["requested"] / GIB
        disk = max(0.0, requested - served)
        x = qi + (di - 0.5) * (width + gap)
        ax2.bar(x, served, width, color=BLUE, edgecolor=SURFACE, linewidth=2)
        ax2.bar(x, disk, width, bottom=served, color=DISK, edgecolor=SURFACE, linewidth=2)
        for h, b, ink in ((served, 0, "white"), (disk, served, "white")):
            if h >= 0.25:
                ax2.text(x, b + h / 2, f"{h:.2f}", ha="center", va="center", fontsize=8.5,
                         color=ink)
        ax2.text(x, -0.1, name, ha="center", va="top", fontsize=8.5, color=MUTED)
ax2.set_xticks(range(3), [f"\n\n{lbl}" for _, lbl in QUERIES], fontsize=9, color=TEXT)
ax2.tick_params(axis="x", length=0)
ax2.set_ylim(0, 3.2)
ax2.set_ylabel("GiB the query reads", color=MUTED, fontsize=9)
ax2.set_title("Where each query's data came from", loc="left", fontsize=11.5, color=TEXT, pad=46)
ax2.legend([Patch(color=BLUE), Patch(color=DISK)], ["Served from GPU cache", "Read from disk"],
           loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8,
           labelcolor=TEXT, handlelength=1.4)

fig.subplots_adjust(left=0.05, right=0.99, top=0.80, bottom=0.15, wspace=0.18)
fig.savefig(HERE / "abc_budget.png", dpi=200, facecolor=SURFACE)
print("wrote", HERE / "abc_budget.png")
