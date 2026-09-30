"""Shared matplotlib style for the paper figures.

Single-column width, Times-compatible serif, thin rules, no chartjunk. Importing
this module applies the rcParams; the palette and helpers are for the figures to
draw with so they stay consistent with each other.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

# Two-step sequential ramp plus a neutral, all legible in greyscale.
NEUTRAL = dict(facecolor="#d8dcda", edgecolor="#2f3b3a", linewidth=0.6)
LIGHT = dict(facecolor="#a8c3bf", edgecolor="#2f3b3a", linewidth=0.6)
DARK = dict(facecolor="#3d6b66", edgecolor="#2f3b3a", linewidth=0.6)
ACCENT = dict(facecolor="#b4522e", edgecolor="#2f3b3a", linewidth=0.6)
GRID = "0.88"
ERRBAR = dict(elinewidth=0.6, capthick=0.6)


def scan_seconds(bucket_csv: Path) -> float:
    """Scan ms for one warm execution of a run's whole query set, in seconds."""
    rows = [r for r in csv.DictReader(bucket_csv.open()) if int(r["execution"]) > 1]
    if not rows:
        raise SystemExit(f"{bucket_csv}: no warm executions")
    n_exec = len({int(r["execution"]) for r in rows})
    return sum(float(r["scan"]) for r in rows) / n_exec / 1000.0


def series(root: Path, tag: str, reps=(1, 2, 3)) -> tuple[float, float] | None:
    """(mean, stdev) across repeats, or None when a condition has no runs."""
    runs = [scan_seconds(root / f"{tag}_{r}" / "bucket.csv")
            for r in reps if (root / f"{tag}_{r}" / "bucket.csv").exists()]
    if not runs:
        return None
    return statistics.mean(runs), (statistics.stdev(runs) if len(runs) > 1 else 0.0)


def finish(ax, ylabel: str | None = None) -> None:
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5)
    ax.tick_params(pad=2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, stem: str) -> None:
    out = Path(__file__).resolve().parents[1] / "experiment" / "figs" / stem
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    print(f"wrote {out}.png / .pdf")
