#!/usr/bin/env python3
"""SF50/3GB result: Baseline vs CELEBI (paging+reorder) total scan/materialize
work, under conditions where the 3GB fixed-page cache budget is actually
exceeded and real LRU eviction occurs (unlike the SF30 result, where
admission control alone was sufficient and eviction never triggered)."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_total(comparison_csv: Path) -> tuple[float, float]:
    with comparison_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    total = next(r for r in rows if r["query"] == "TOTAL")
    return float(total["baseline_scan"]) / 1000.0, float(total["proposed_scan"]) / 1000.0


def main() -> int:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"

    comparison_csv = REPO_ROOT / "experiment" / "fixed_page_runs" / "sf50_3gb_evict_check_5x" / "comparison.csv"
    base_s, prop_s = read_total(comparison_csv)
    pct = (base_s - prop_s) / base_s * 100.0

    color_base = "#2a78d6"
    color_prop = "#eb6834"
    ink_primary = "#0b0b0b"

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    bars = ax.bar(["Baseline\n(no paging)", "CELEBI\n(paging + reorder)"], [base_s, prop_s],
                   width=0.55, color=[color_base, color_prop])
    for rect, v in zip(bars, [base_s, prop_s]):
        ax.annotate(f"{v:.0f}s", (rect.get_x() + rect.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 8), fontsize=20, color=ink_primary, ha="center")

    ax.annotate(f"-{pct:.1f}%", (0.5, max(base_s, prop_s) * 0.55), xycoords=("axes fraction", "data"),
                fontsize=24, color=color_prop, ha="center", fontweight="bold")

    ax.set_ylabel("Total scan/materialize work (s)", fontsize=18, color=ink_primary)
    ax.set_title("SF50, 3GB cache budget\n(5 evictions, 68 pages evicted)", fontsize=15, color=ink_primary)
    ax.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=16)
    ax.set_ylim(0, max(base_s, prop_s) * 1.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_png = REPO_ROOT / "experiment" / "graph" / "sf50_3gb_baseline_vs_celebi.png"
    out_pdf = REPO_ROOT / "experiment" / "graph" / "sf50_3gb_baseline_vs_celebi.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
