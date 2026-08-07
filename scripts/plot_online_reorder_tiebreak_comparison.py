#!/usr/bin/env python3
"""Online resident-aware reorder: baseline vs tiebreak variants (future/previous/
smallest/largest column count), scan latency and event-count cache hit rate.

Sources (all random22, SF50, 6GB budget, 10 executions / 1 warmup dropped,
online resident-aware reordering, same methodology as random22_breakdown_sf50_10x):
  experiment/random22_breakdown_sf50_10x_online_resident/comparison_celebi_online.csv       (future)
  experiment/random22_breakdown_sf50_10x_online_previous/comparison_celebi_online.csv       (previous)
  experiment/random22_breakdown_sf50_10x_online_smallest/comparison_celebi_online.csv       (smallest)
  experiment/random22_breakdown_sf50_10x_online_resident_v2/comparison_celebi_online.csv    (largest)
Event-count hit rates are hardcoded from the matching analysis run in-session
(future/previous: backed=288 miss=1224 excluded=2385; smallest/largest:
backed=165 miss=1558 excluded=2385)."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = [
    ("future", "future_overlap\n(original)", "experiment/random22_breakdown_sf50_10x_online_resident/comparison_celebi_online.csv", 19.05),
    ("previous", "previous_overlap\n(O(1))", "experiment/random22_breakdown_sf50_10x_online_previous/comparison_celebi_online.csv", 19.05),
    ("smallest", "fewest\ncolumns first", "experiment/random22_breakdown_sf50_10x_online_smallest/comparison_celebi_online.csv", 9.58),
    ("largest", "most\ncolumns first", "experiment/random22_breakdown_sf50_10x_online_resident_v2/comparison_celebi_online.csv", 9.58),
]


def read_total_scan(comparison_csv: Path) -> tuple[float, float]:
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

    color_baseline = "#8a8a86"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    ink_primary = "#0b0b0b"
    ink_secondary = "#52514e"

    baseline_s = None
    scan_vals = []
    hit_vals = []
    labels = []
    for _, label, rel_path, hit_rate in CONDITIONS:
        b, p = read_total_scan(REPO_ROOT / rel_path)
        baseline_s = b
        scan_vals.append(p)
        hit_vals.append(hit_rate)
        labels.append(label)

    fig, (ax_scan, ax_hit) = plt.subplots(1, 2, figsize=(13, 6.5))

    # --- left: scan latency ---
    bar_labels = ["Baseline\n(no caching)"] + labels
    bar_vals = [baseline_s] + scan_vals
    bar_colors = [color_baseline] + colors
    bars = ax_scan.bar(bar_labels, bar_vals, width=0.6, color=bar_colors)
    for rect, v in zip(bars, bar_vals):
        ax_scan.annotate(f"{v:.0f}s", (rect.get_x() + rect.get_width() / 2, v),
                          textcoords="offset points", xytext=(0, 6), fontsize=12,
                          color=ink_primary, ha="center")
    for rect, v in zip(bars[1:], scan_vals):
        pct = (baseline_s - v) / baseline_s * 100.0
        ax_scan.annotate(f"-{pct:.1f}%", (rect.get_x() + rect.get_width() / 2, v * 0.5),
                          fontsize=11, color="white", ha="center", fontweight="bold")

    # previous_overlap is bars[2] (baseline, future, previous, smallest, largest);
    # reorder cost measured in-session via run_online_resident_aware_reorder.py's
    # perf_counter instrumentation, 3 executions x 21 decisions each of the same
    # 22-query random arrival: rank_decision_total=5.05ms + log_parse_total=67.47ms
    # over 63 decisions = 72.52ms / 3 executions = ~24.2ms per 22-query workload.
    previous_bar = bars[2]
    ax_scan.annotate(
        "reorder cost\n~24ms/22-query run",
        (previous_bar.get_x() + previous_bar.get_width() / 2, 0),
        textcoords="offset points", xytext=(0, -38), fontsize=8.5,
        color=ink_primary, ha="center", va="top",
    )

    ax_scan.set_ylabel("Total scan latency (s, hot 9-exec avg)", fontsize=13, color=ink_primary)
    ax_scan.set_title("Online resident-aware reorder: scan latency", fontsize=13, color=ink_primary)
    ax_scan.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=10)
    ax_scan.set_ylim(0, baseline_s * 1.2)
    for spine in ("top", "right"):
        ax_scan.spines[spine].set_visible(False)

    # --- right: event-count cache hit rate ---
    bars2 = ax_hit.bar(labels, hit_vals, width=0.6, color=colors)
    for rect, v in zip(bars2, hit_vals):
        ax_hit.annotate(f"{v:.2f}%", (rect.get_x() + rect.get_width() / 2, v),
                         textcoords="offset points", xytext=(0, 6), fontsize=12,
                         color=ink_primary, ha="center")
    ax_hit.set_ylabel("Event-count cache hit rate (%)", fontsize=13, color=ink_primary)
    ax_hit.set_title("Cache hit rate by tiebreak rule", fontsize=13, color=ink_primary)
    ax_hit.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=10)
    ax_hit.set_ylim(0, max(hit_vals) * 1.35)
    for spine in ("top", "right"):
        ax_hit.spines[spine].set_visible(False)

    fig.suptitle("SF50, 6GB budget, 22-query random arrival, 10 executions (1 warmup dropped)",
                 fontsize=11, color=ink_secondary, y=1.02)
    fig.tight_layout()

    out_png = REPO_ROOT / "experiment" / "graph" / "online_reorder_tiebreak_comparison.png"
    out_pdf = REPO_ROOT / "experiment" / "graph" / "online_reorder_tiebreak_comparison.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
