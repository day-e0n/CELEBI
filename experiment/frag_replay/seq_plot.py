"""Eight queries run twice: the 6 GiB cache budget after each query, and where each query's
data came from, for batch-granular caching and uniform 16 MiB pages.

Inputs: seq_meta.json (seq_trace.py), seq_<design>.csv and seq_resident_<design>.txt (replay.cu).

  pixi run -e duckdb-python python experiment/frag_replay/seq_plot.py
"""
import csv
import json
from collections import Counter
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
LETTERS = "ABCDEFGH"
KEY = ("A q16 UserID · B q17 +SearchPhrase · C q19 +EventTime · D q9 RegionID, UserID · "
       "E q10 +AdvEngineID, ResolutionWidth · F q33 WatchID, ClientIP, IsRefresh, ResolutionWidth · "
       "G q34 URL (8.8 GiB) · H q36 ClientIP")

meta = json.load(open(HERE / "seq_meta.json"))
steps = meta["steps"]


def composition(design):
    rows = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(open(HERE / f"seq_{design}.csv"))]
    resident = [list(map(int, l.split())) for l in open(HERE / f"seq_resident_{design}.txt")]
    entries = meta["entries"][design]
    out = []
    for m, ids in zip(rows, resident):
        units = Counter()
        size = {}
        for i in ids:
            for c, g, b in entries[str(i)]:
                units[(c, g)] += 1
                size[(c, g)] = b
        unique = sum(size[u] for u in units)
        held = m["live_payload"]
        free_total = m["budget"] - m["pool_used"]
        out.append({"unique": unique, "duplicate": max(0, held - unique),
                    "intfrag": m["live_requested"] - held,
                    "extfrag": max(0, free_total - m["largest_free"]),
                    "free": min(free_total, m["largest_free"]),
                    "served_cum": m["served_bytes"]})
    return out


comp = {d: composition(d) for d, _ in DESIGNS}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbcb6")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_facecolor(SURFACE)


n = len(steps)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8.6), facecolor=SURFACE, sharex=True,
                               gridspec_kw={"height_ratios": [1.35, 1]})
style(ax1)
style(ax2)
width, gap = 0.38, 0.04
for si in range(n):
    for di, (d, name) in enumerate(DESIGNS):
        c = comp[d][si]
        x = si + (di - 0.5) * (width + gap)
        bottom = 0.0
        for h, color, hatch in ((c["unique"], BLUE, None), (c["duplicate"], BLUE, "///"),
                                (c["intfrag"], YELLOW, None), (c["extfrag"], VIOLET, None),
                                (c["free"], FREE, None)):
            h /= GIB
            if h > 0:
                ax1.bar(x, h, width, bottom=bottom, color=color, hatch=hatch, edgecolor=SURFACE,
                        linewidth=1.2)
            bottom += h
        served = (c["served_cum"] - (comp[d][si - 1]["served_cum"] if si else 0)) / GIB
        disk = max(0.0, steps[si]["requested"] / GIB - served)
        ax2.bar(x, served, width, color=BLUE, edgecolor=SURFACE, linewidth=1.2)
        ax2.bar(x, disk, width, bottom=served, color=DISK, edgecolor=SURFACE, linewidth=1.2)
        if served >= 0.6:
            ax2.text(x, served / 2, f"{served:.1f}", ha="center", va="center", fontsize=7.5,
                     color="white", rotation=90)
        ax2.text(x, -0.25, "b" if d == "batch" else "p", ha="center", va="top", fontsize=7.5,
                 color=MUTED)

ax1.set_ylim(0, 6.0)
ax1.set_ylabel("GiB of the 6 GiB cache budget", color=MUTED, fontsize=9)
ax1.set_title("Cache budget after each query  (left bar: batch, right bar: 16 MiB page)",
              loc="left", fontsize=11.5, color=TEXT, pad=40)
ax1.legend([Patch(color=BLUE), Patch(facecolor=BLUE, hatch="///", edgecolor=SURFACE),
            Patch(color=YELLOW), Patch(color=VIOLET), Patch(color=FREE)],
           ["Cached data", "Duplicate copy of cached data", "Internal fragmentation (padding)",
            "External fragmentation (free, unusable)", "Free, usable"],
           loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5, frameon=False, fontsize=8.5,
           labelcolor=TEXT, handlelength=1.4)
ax1.axvline(7.5, color="#bdbcb6", linewidth=1, linestyle="--")

ax2.set_ylim(0, 9.4)
ax2.set_ylabel("GiB the query reads", color=MUTED, fontsize=9)
ax2.set_title("Where each query's data came from  (b: batch, p: 16 MiB page)", loc="left",
              fontsize=11.5, color=TEXT, pad=26)
ax2.legend([Patch(color=BLUE), Patch(color=DISK)], ["Served from GPU cache", "Read from disk"],
           loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5,
           labelcolor=TEXT, handlelength=1.4)
ax2.axvline(7.5, color="#bdbcb6", linewidth=1, linestyle="--")
ax2.set_xticks(range(n), [f"\n{LETTERS[i % 8]}" for i in range(n)], fontsize=10, color=TEXT)
ax2.tick_params(axis="x", length=0)
for r, x0 in ((1, 3.5), (2, 11.5)):
    ax2.text(x0, -1.6, f"round {r}", ha="center", va="top", fontsize=9.5, color=TEXT)
fig.text(0.05, 0.015, KEY, fontsize=8, color=MUTED)

totals = {d: comp[d][-1]["served_cum"] / GIB for d, _ in DESIGNS}
asked = sum(s["requested"] for s in steps) / GIB
fig.text(0.99, 0.965, f"served from cache over both rounds: batch {totals['batch']:.1f} GiB, "
         f"16 MiB page {totals['page']:.1f} GiB of {asked:.1f} GiB read",
         ha="right", fontsize=9, color=TEXT)
fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.12, hspace=0.42)
fig.savefig(HERE / "seq_budget.png", dpi=200, facecolor=SURFACE)

with open(HERE / "seq_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "query", "design", "unique_gib", "duplicate_gib", "intfrag_gib",
                "extfrag_gib", "free_gib", "served_gib", "read_gib"])
    for d, _ in DESIGNS:
        for si in range(n):
            c = comp[d][si]
            served = (c["served_cum"] - (comp[d][si - 1]["served_cum"] if si else 0)) / GIB
            w.writerow([si, LETTERS[si % 8] + str(si // 8 + 1), d] +
                       [f"{c[k] / GIB:.2f}" for k in ("unique", "duplicate", "intfrag", "extfrag", "free")] +
                       [f"{served:.2f}", f"{steps[si]['requested'] / GIB:.2f}"])
print("wrote seq_budget.png, seq_summary.csv")
