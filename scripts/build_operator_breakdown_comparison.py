#!/usr/bin/env python3
"""Combine baseline (cold_hot) vs proposed (paging+reorder) operator-bucket CSVs
(as produced by parse_quent_operator_breakdown.py) into the paper comparison:
per-query and workload-total scan/join/aggregate/filter/sort/other GPU work time,
baseline vs proposed, with % change per bucket (headline: scan decline).

Series rule mirrors run_fixed_page_workload_sequence.py's series_for(): the first
execution of each condition is warmup (cold / paging_warmup) and excluded; the
remaining executions are averaged as the condition's steady-state series
(hot for cold_hot, paging for paging).
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

BUCKETS = ("scan", "join", "aggregate", "filter", "sort", "other")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return 0.0


def steady_state_series(condition: str, execution: int) -> str | None:
    if condition == "cold_hot":
        return None if execution == 1 else "hot"
    if condition == "paging":
        return None if execution == 1 else "paging"
    return condition


def aggregate_by_query(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """query -> bucket -> mean ms across included (steady-state) executions."""

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        series = steady_state_series(row["condition"], int(row["execution"]))
        if series is None:
            continue
        query = row["query"]
        for b in (*BUCKETS, "total_ms"):
            grouped[query][b].append(to_float(row[b]))
    out: dict[str, dict[str, float]] = {}
    for query, buckets in grouped.items():
        out[query] = {b: statistics.mean(vals) if vals else 0.0 for b, vals in buckets.items()}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bucket-csv", type=Path, required=True)
    parser.add_argument("--proposed-bucket-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, default=None)
    parser.add_argument("--out-pdf", type=Path, default=None)
    parser.add_argument("--proposed-label", default="Baseline+page caching")
    args = parser.parse_args()

    baseline = aggregate_by_query(read_rows(args.baseline_bucket_csv))
    proposed = aggregate_by_query(read_rows(args.proposed_bucket_csv))
    queries = sorted(set(baseline) & set(proposed), key=lambda q: int(q[1:]))

    fields = ["query"]
    for b in (*BUCKETS, "total_ms"):
        fields += [f"baseline_{b}", f"proposed_{b}", f"{b}_change_pct"]
    rows_out = []
    totals_baseline = defaultdict(float)
    totals_proposed = defaultdict(float)
    for q in queries:
        row = {"query": q}
        for b in (*BUCKETS, "total_ms"):
            base_v = baseline[q][b]
            prop_v = proposed[q][b]
            totals_baseline[b] += base_v
            totals_proposed[b] += prop_v
            pct = (base_v - prop_v) / base_v * 100.0 if base_v else 0.0
            row[f"baseline_{b}"] = base_v
            row[f"proposed_{b}"] = prop_v
            row[f"{b}_change_pct"] = pct
        rows_out.append(row)

    totals_row = {"query": "TOTAL"}
    for b in (*BUCKETS, "total_ms"):
        base_v = totals_baseline[b]
        prop_v = totals_proposed[b]
        pct = (base_v - prop_v) / base_v * 100.0 if base_v else 0.0
        totals_row[f"baseline_{b}"] = base_v
        totals_row[f"proposed_{b}"] = prop_v
        totals_row[f"{b}_change_pct"] = pct
    rows_out.append(totals_row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)
    print(f"wrote {args.out_csv}")
    print(
        f"TOTAL scan: baseline={totals_baseline['scan']:.1f}ms proposed={totals_proposed['scan']:.1f}ms "
        f"({totals_row['scan_change_pct']:.1f}% decline)"
    )

    if args.out_png or args.out_pdf:
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np

        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        matplotlib.rcParams["font.family"] = "sans-serif"

        # Validated categorical palette (dataviz skill, references/palette.md), slots
        # 1-6 in fixed order — passes CVD/normal-vision adjacent-pair checks for a
        # 6-series stacked bar. Text/labels use ink tokens below, never these hues.
        colors = {
            "scan": "#2a78d6",       # slot 1 blue
            "join": "#eb6834",       # slot 2 orange
            "aggregate": "#1baf7a",  # slot 3 aqua
            "filter": "#eda100",     # slot 4 yellow
            "sort": "#e87ba4",       # slot 5 magenta
            "other": "#008300",      # slot 6 green
        }
        ink_primary = "#000000"

        fig, ax = plt.subplots(figsize=(max(19, len(queries) * 0.95), 7.0))
        x = np.arange(len(queries))
        pair_width = 0.34
        hatches = {"baseline": None, "proposed": "///"}
        for offset, (label, data) in zip((-1, 1), (("baseline", baseline), ("proposed", proposed))):
            bottoms = np.zeros(len(queries))
            bar_x = x + offset * (pair_width / 2)  # bars of a pair touch (no gap); hatch tells them apart
            for b in BUCKETS:
                vals = np.array([data[q][b] for q in queries])
                ax.bar(
                    bar_x,
                    vals,
                    bottom=bottoms,
                    width=pair_width,
                    color=colors[b],
                    label=b if offset == -1 else None,
                    edgecolor="black",
                    linewidth=0.8,
                    hatch=hatches[label],
                )
                bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(queries, rotation=0, fontsize=20, color=ink_primary)
        ax.set_xlabel("TPC-H query", fontsize=22, color=ink_primary)
        ax.set_ylabel("GPU operator work time (ms)", fontsize=22, color=ink_primary)
        ax.tick_params(axis="y", labelsize=20, colors=ink_primary)
        ax.tick_params(axis="x", colors=ink_primary)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.2)
        ax.set_xlim(x[0] - 0.75, x[-1] + 0.75)

        # Plain paper-figure layout: no in-image headline/subtitle (that belongs in the
        # LaTeX caption/body text) — legend lives in its own boxed panel fully above
        # the axes, not overlapping the plot. Wide flat swatches (long rectangles,
        # not squares) for both the operator-color legend and the solid/hatch key.
        from matplotlib.patches import Patch

        fig.subplots_adjust(top=0.83, left=0.05, right=0.99, bottom=0.08)
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncols=6,
            frameon=True,
            fontsize=19,
            labelcolor=ink_primary,
            handlelength=3.2,
            handleheight=0.9,
            columnspacing=1.6,
            edgecolor="black",
            fancybox=False,
            borderpad=0.9,
        )
        legend.get_frame().set_linewidth(1.2)

        style_handles = [
            Patch(facecolor="white", edgecolor="black", linewidth=0.8, label="Baseline"),
            Patch(facecolor="white", edgecolor="black", linewidth=0.8, hatch="///",
                  label=args.proposed_label),
        ]
        style_legend = ax.legend(
            handles=style_handles,
            loc="upper left",
            ncols=1,
            frameon=True,
            fontsize=16,
            labelcolor=ink_primary,
            handlelength=3.2,
            handleheight=0.9,
            labelspacing=0.5,
            edgecolor="black",
            fancybox=False,
            borderpad=0.9,
        )
        style_legend.get_frame().set_linewidth(1.2)
        ax.add_artist(style_legend)

        args_out = [p for p in (args.out_png, args.out_pdf) if p is not None]
        for out_path in args_out:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
            print(f"wrote {out_path}")
        plt.close(fig)

        # Companion figure: total work time per operator on a log axis, so the
        # operators that are invisible slivers in the stacked per-query chart
        # (filter, sort) still show up as their own bars.
        fig2, ax2 = plt.subplots(figsize=(10, 6.5))
        xb = np.arange(len(BUCKETS))
        bw = 0.36
        for offset, (label, totals) in zip((-1, 1), (("Baseline", totals_baseline), ("CELEBI", totals_proposed))):
            vals = np.array([totals[b] for b in BUCKETS])
            bars2 = ax2.bar(
                xb + offset * bw / 2,
                vals,
                width=bw,
                color=[colors[b] for b in BUCKETS],
                edgecolor="black",
                linewidth=0.8,
                hatch="///" if offset == 1 else None,
            )
            for rect, v in zip(bars2, vals):
                ax2.text(
                    rect.get_x() + rect.get_width() / 2, v * 1.15, f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=12, color=ink_primary, rotation=90,
                )

        ax2.set_yscale("log")
        ax2.set_xticks(xb)
        ax2.set_xticklabels(BUCKETS, fontsize=16, color=ink_primary)
        ax2.set_ylabel("Total GPU operator work time (ms, log scale)", fontsize=15, color=ink_primary)
        ax2.tick_params(axis="y", labelsize=14, colors=ink_primary)
        ax2.set_ylim(top=ax2.get_ylim()[1] * 6)
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.2)
        ax2_legend = ax2.legend(
            handles=style_handles, loc="upper right", frameon=True, fontsize=14,
            labelcolor=ink_primary, handlelength=2.6, handleheight=0.9, edgecolor="black", fancybox=False,
        )
        ax2_legend.get_frame().set_linewidth(1.0)
        fig2.tight_layout()

        totals_out = [
            Path(str(p.with_suffix("")) + "_by_operator_log" + p.suffix) for p in args_out
        ]
        for out_path in totals_out:
            fig2.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
            print(f"wrote {out_path}")
        plt.close(fig2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
