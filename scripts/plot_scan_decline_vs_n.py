#!/usr/bin/env python3
"""Real-execution scaling result: total GPU scan-time decline (%) of paging+
reorder vs baseline, across batch size N (built from real qgen-varied
instances, 3GB page cache budget, 3 executions each condition)."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_total_scan_pct(n: int) -> float:
    path = REPO_ROOT / "experiment" / "expB_scaled_real" / f"N{n}" / "comparison.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    total = next(r for r in rows if r["query"] == "TOTAL")
    return float(total["scan_change_pct"])


def main() -> int:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"

    ns = [10, 20, 30, 40, 50, 60]
    pct = [read_total_scan_pct(n) for n in ns]

    color = "#2a78d6"
    ink_primary = "#0b0b0b"
    gridline = "#d8d7d2"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ns, pct, color=color, linewidth=2.8, marker="o", markersize=10)
    for n, p in zip(ns, pct):
        ax.annotate(f"{p:.1f}%", (n, p), textcoords="offset points", xytext=(0, 14),
                    fontsize=16, color=ink_primary, ha="center")

    ax.set_xlabel("batch size N", fontsize=18, color=ink_primary)
    ax.set_ylabel("total scan-time decline (%)", fontsize=18, color=ink_primary)
    ax.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    ax.set_xticks(ns)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(gridline)
    ax.set_ylim(0, max(pct) * 1.25)

    fig.tight_layout()
    out_png = REPO_ROOT / "experiment" / "graph" / "expB_scan_decline_vs_n.png"
    out_pdf = REPO_ROOT / "experiment" / "graph" / "expB_scan_decline_vs_n.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
