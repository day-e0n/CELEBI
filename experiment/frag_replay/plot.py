"""How a 6 GiB GPU cache budget is spent under three cache granularities.

Inputs are the replay CSVs (replay.cu over traces from gen_trace.py): one row per scan of
the ClickBench baseline run (41 queries x 3 executions), measured on a cudaMallocAsync pool
capped at the budget.

Two versions of each figure:
  *_reuse.png   cached data split into reused / not (yet) reused
  *.png         cached data as one category

  pixi run -e duckdb-python python experiment/frag_replay/plot.py
"""
from pathlib import Path

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GIB = 2 ** 30
WARMUP = 50  # scans before the cache first fills; excluded from the averages

POLICIES = [("batch", "Batch granularity\n(one allocation per batch x column)"),
            ("page", "Exact-size pages\n(<=16 MiB, data/mask/offsets separate)"),
            ("slab", "Uniform 16 MiB pages\n(one allocation each, padded)")]
SHORT = {"batch": "Batch", "page": "Exact-size pages", "slab": "Uniform 16 MiB pages"}

# Bottom to top. Same hues as gpu_cache_budget.html; validated adjacent pairs on light.
WASTE_AND_FREE = [("intfrag", "Internal fragmentation (padding)", "#eda100"),
                  ("extfrag", "External fragmentation (free, unusable)", "#4a3aa7"),
                  ("free", "Free, usable", "#e4e2dc")]
VARIANTS = {
    "_reuse": [("reused", "Cached data, reused", "#2a78d6"),
               ("unreused", "Cached data, not (yet) reused", "#1baf7a")] + WASTE_AND_FREE,
    "": [("cached", "Cached data", "#2a78d6")] + WASTE_AND_FREE,
}
WHITE_INK = {"reused", "unreused", "cached", "extfrag"}
TEXT, MUTED, SURFACE = "#16161a", "#5a5a60", "#fbfbf9"


def load(policy):
    rows = list(csv.DictReader(open(HERE / f"{policy}.csv")))
    out = []
    for r in rows:
        v = {k: int(r[k]) for k in r}
        free_total = v["budget"] - v["pool_used"]
        out.append({
            "scan": v["scan"],
            "reused": v["reused_payload"],
            "unreused": v["live_payload"] - v["reused_payload"],
            "cached": v["live_payload"],
            "intfrag": v["live_requested"] - v["live_payload"],
            "extfrag": max(0, free_total - v["largest_free"]),
            "free": min(free_total, v["largest_free"]),
        })
    return out


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbcb6")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_facecolor(SURFACE)


data = {p: load(p) for p, _ in POLICIES}

for suffix, cats in VARIANTS.items():
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in cats]
    names = [n for _, n, _ in cats]

    # ---- Figure 1: average composition of the budget ---------------------------------
    fig, ax = plt.subplots(figsize=(11, 3.9), facecolor=SURFACE)
    style(ax)
    for i, (p, _) in enumerate(POLICIES):
        rows = data[p][WARMUP:]
        left = 0.0
        for k, _, color in cats:
            w = sum(r[k] for r in rows) / len(rows) / GIB
            ax.barh(i, w, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
            if w >= 0.32:
                ink = "#ffffff" if k in WHITE_INK else TEXT
                ax.text(left + w / 2, i, f"{w:.2f}", ha="center", va="center", fontsize=9,
                        color=ink)
            left += w
    ax.set_yticks(range(len(POLICIES)), [l for _, l in POLICIES], fontsize=9, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, 6.0)
    ax.set_xlabel("GiB of the 6 GiB cache budget (average after warm-up)", color=MUTED,
                  fontsize=9)
    ax.set_title("Where the 6 GiB GPU cache budget goes", loc="left", fontsize=12,
                 color=TEXT, pad=34)
    ax.legend(handles, names, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=3 if len(cats) > 4 else 4, frameon=False, fontsize=8.5, labelcolor=TEXT,
              handlelength=1.2)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.76, bottom=0.14)
    fig.savefig(HERE / f"budget_composition{suffix}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # ---- Figure 2: the same composition over the run ---------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True, facecolor=SURFACE)
    for ax, (p, _) in zip(axes, POLICIES):
        style(ax)
        rows = data[p]
        xs = [r["scan"] for r in rows]
        ys = [[r[k] / GIB for r in rows] for k, _, _ in cats]
        ax.stackplot(xs, ys, colors=[c for _, _, c in cats], linewidth=0)
        ax.set_title(SHORT[p], loc="left", fontsize=10.5, color=TEXT)
        ax.set_xlim(0, xs[-1])
        ax.set_ylim(0, 6.0)
        ax.set_xlabel("scan (41 queries x 3 runs)", color=MUTED, fontsize=9)
    axes[0].set_ylabel("GiB", color=MUTED, fontsize=9)
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(0.045, 1.0),
               ncol=len(cats), frameon=False, fontsize=8.5, labelcolor=TEXT, handlelength=1.2)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.78, bottom=0.15, wspace=0.08)
    fig.savefig(HERE / f"budget_over_time{suffix}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)

# ---- Table view ----------------------------------------------------------------------
keys = ["reused", "unreused", "cached", "intfrag", "extfrag", "free"]
with open(HERE / "summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["policy"] + [f"{k}_gib" for k in keys])
    for p, _ in POLICIES:
        rows = data[p][WARMUP:]
        w.writerow([p] + [f"{sum(r[k] for r in rows) / len(rows) / GIB:.3f}" for k in keys])
print(open(HERE / "summary.csv").read())
