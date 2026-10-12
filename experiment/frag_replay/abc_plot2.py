"""Three queries, batch cache vs page cache: where each query's data came from, and what the
cache holds afterwards. A simpler companion to abc_plot.py.

Query 1 = ClickBench q16 (UserID), Query 2 = q17 (UserID, SearchPhrase),
Query 3 = q19 (UserID, SearchPhrase, EventTime), as run on Sirius (experiment/ABC_base).

  pixi run -e duckdb-python python experiment/frag_replay/abc_plot2.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from gen_trace import buffers, rg_parts, rg_rows

HERE = Path(__file__).resolve().parent
GIB = 2 ** 30
TEXT, MUTED, SURFACE = "#16161a", "#5a5a60", "#fbfbf9"
CACHE, DISK, PAD = "#008300", "#d9d8d4", "#e4e2dc"
COLOR = {"UserID": "#2a78d6", "SearchPhrase": "#eb6834", "EventTime": "#4a3aa7"}
QUERIES = [("Query 1", ["UserID"]),
           ("Query 2", ["UserID", "SearchPhrase"]),
           ("Query 3", ["UserID", "SearchPhrase", "EventTime"])]

col_gib = {c: sum(sum(buffers(c, rg_rows[g], rg_parts(c, g)[1])) for g in rg_rows) / GIB
           for c in COLOR}
meas = {p: [{k: int(v) for k, v in r.items()} for r in csv.DictReader(open(HERE / f"abc_{p}.csv"))]
        for p in ("batch", "page")}
content = json.load(open(HERE / "abc_content.json"))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbcb6")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_facecolor(SURFACE)


fig, (top, bot) = plt.subplots(2, 1, figsize=(11.5, 7.6), facecolor=SURFACE,
                               gridspec_kw={"height_ratios": [1.15, 1]})
style(top)
style(bot)

# ---- Top: where each query's data came from -------------------------------------------
width, gap = 0.34, 0.06
for qi, (qname, cols) in enumerate(QUERIES):
    asked = sum(col_gib[c] for c in cols)
    for di, p in enumerate(("batch", "page")):
        served_cum = [r["served_bytes"] / GIB for r in meas[p]]
        served = served_cum[qi] - (served_cum[qi - 1] if qi else 0.0)
        disk = max(0.0, asked - served)
        x = qi + (di - 0.5) * (width + gap)
        top.bar(x, served, width, color=CACHE, edgecolor=SURFACE, linewidth=2)
        top.bar(x, disk, width, bottom=served, color=DISK, edgecolor=SURFACE, linewidth=2)
        if served > 0.05:
            top.text(x, served / 2, f"{served:.2f}", ha="center", va="center", fontsize=9,
                     color="white")
        if disk > 0.05:
            top.text(x, served + disk / 2, f"{disk:.2f}", ha="center", va="center", fontsize=9,
                     color=TEXT)
        top.text(x, -0.07, "batch\ncache" if p == "batch" else "page\ncache", ha="center",
                 va="top", fontsize=8.5, color=MUTED, linespacing=1.1)
top.set_xticks(range(3), [f"\n\n\n{q}: reads {', '.join(c)}" for q, c in QUERIES], fontsize=9.5,
               color=TEXT)
top.tick_params(axis="x", length=0)
top.set_ylim(0, 2.9)
top.set_ylabel("GiB", color=MUTED, fontsize=9)
top.set_title("Where each query's data came from", loc="left", fontsize=12, color=TEXT, pad=28)
top.legend([Patch(color=CACHE), Patch(color=DISK)], ["From the GPU cache", "From disk"],
           loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9,
           labelcolor=TEXT, handlelength=1.3)

# ---- Bottom: what the cache holds after Query 3 --------------------------------------
rows = [("Batch cache", [(c, q) for q, cols in QUERIES for c in cols], 0.0),
        ("Page cache\n(16 MiB pages)", [(c, QUERIES[[c in cols for _, cols in QUERIES].index(True)][0])
                                       for c in COLOR],
         (meas["page"][-1]["live_requested"] - meas["page"][-1]["live_payload"]) / GIB)]
for ri, (name, segs, padding) in enumerate(rows):
    left = 0.0
    for col, q in segs:
        w = col_gib[col]
        bot.barh(ri, w, left=left, height=0.56, color=COLOR[col], edgecolor=SURFACE, linewidth=2)
        bot.text(left + w / 2, ri, f"{col}\n(stored by {q})", ha="center", va="center",
                 fontsize=8, color="white", linespacing=1.15)
        left += w
    if padding > 0:
        bot.barh(ri, padding, left=left, height=0.56, color=PAD, edgecolor=SURFACE, linewidth=2)
        left += padding
    note = f"{left:.2f} GiB" + (f"\nincl. {padding:.2f} GiB padding" if padding else
                                "\nUserID 3 times,\nSearchPhrase 2 times")
    bot.text(left + 0.06, ri, note, va="center", fontsize=9, color=TEXT)
bot.set_yticks(range(2), [r[0] for r in rows], fontsize=9.5, color=TEXT)
bot.invert_yaxis()
bot.set_xlim(0, 6.0)
bot.set_xlabel("GiB of the 6 GiB cache budget", color=MUTED, fontsize=9)
bot.set_title("What the cache holds after Query 3", loc="left", fontsize=12, color=TEXT, pad=10)

fig.text(0.01, 0.012,
         "ClickBench, 100M rows. Query 1, 2, 3 = q16, q17, q19, run in that order on Sirius; same "
         "data batches and a 6 GiB cache for both.\nThe batch cache stores each data batch whole; "
         "the page cache stores 16 MiB pages per column and row group.",
         fontsize=8, color=MUTED)
fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.14, hspace=0.62)
fig.savefig(HERE / "abc_simple.png", dpi=200, facecolor=SURFACE)
print("wrote abc_simple.png", {c: round(v, 2) for c, v in col_gib.items()})
