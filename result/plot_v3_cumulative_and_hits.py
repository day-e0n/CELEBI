#!/usr/bin/env python3
"""Current-engine figures in the headline's style (scripts/paper_style.py).

  cumulative_query_time_v3_styled   query time accumulated q1..qN in query-number order,
                             baseline / paging / paging + reorder; the last point is the
                             workload total (= headline_cache_reorder_v3). experiment/A_*,
                             TPC-H 3 runs, ClickBench 2, executions 2-3.
  per_query_hit_v3           per query, bytes the page cache served / decoded bytes its scans
                             handed downstream. experiment/HIT2_* (SIRIUS_CACHE_HIT_TRACE=1),
                             1 run, executions 2-3.
  cache_hit_summary_v3       the same ratio over the whole workload.
Each query's own value sits at its query number wherever the reordered batch ran it. "paging +
reorder" is the reordered batch with order-aware eviction. Engine: late materialization on,
scan no-history reservation 3x.

  pixi run -e duckdb-python python result/plot_v3_cumulative_and_hits.py
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
EXP = ROOT / "experiment"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(OUT))
import paper_style as ps  # noqa: E402
import cache_hit_rate as chr_  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CONDS = [("baseline", "base", "#8f9a98", "o"), ("paging", "lru", "#3d6b66", "s"),
         ("paging + reorder", "best", "#b4522e", "D")]
# Cumulative figure: colours and names of the earlier CELEBI figures
# (scripts/plot_hitrate_and_scanwork_combined.py).
CUM_CONDS = [("Baseline", "base", "#4c78a8"), ("Baseline+page caching", "lru", "#59a14f"),
             ("Baseline+page caching+reordering (CELEBI)", "best", "#f2b134")]
WORKLOADS = [("ClickBench", "cb", (1, 2), [q for q in range(1, 44) if q not in (28, 29)]),
             ("TPC-H SF50", "th", (1, 2, 3), list(range(1, 23)))]


def wall_by_query(run: str) -> dict[int, float]:
    """Per query, end-to-end wall averaged over executions 2-3."""
    per: dict[int, list[float]] = {}
    for line in open(EXP / run / "runner.log"):
        m = re.search(r"exec=(\d+)/\d+ .* q(\d+) done wall=([\d.]+)s", line)
        if m and int(m[1]) >= 2:
            per.setdefault(int(m[2]), []).append(float(m[3]))
    return {q: statistics.mean(v) for q, v in per.items()}


def hit_by_query(bench: str, cond: str) -> tuple[dict[int, float], float]:
    data = chr_.collect(f"HIT2_{bench}_{cond}")
    per = {q: chr_.summarize(c for (q2, _), c in data.items() if q2 == q)["byte_hit_pct"]
           for q in {q for q, _ in data}}
    return per, chr_.summarize(data.values())["byte_hit_pct"]


def style_axis(ax, labels, ylabel):
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=5.2 if len(labels) > 25 else 6.5)
    ax.set_xlim(0.3, len(labels) + 0.7)
    ps.finish(ax, ylabel)


def cumulative() -> None:
    # Match the compact sans-serif reference; keep styling local to this figure.
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.linewidth": 0.55, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "savefig.bbox": None}):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9),
                                 gridspec_kw={"width_ratios": [1, 1]})
        fig.subplots_adjust(left=0.065, right=0.935, bottom=0.20, top=0.79, wspace=0.25)
        exported = []
        for ax, (title, bench, reps, queries) in zip(axes, WORKLOADS):
            n = len(queries)
            finals = []
            for label, cond, color in CUM_CONDS:
                walls = [wall_by_query(f"A_{bench}_{cond}_r{r}") for r in reps]
                total, cum = 0.0, []
                for position, q in enumerate(queries, 1):
                    wall = statistics.mean(w[q] for w in walls)
                    total += wall
                    cum.append(total)
                    exported.append({"workload": title, "condition": label, "query": f"q{q}",
                                     "position": position, "wall_s": wall, "cumulative_wall_s": total})
                finals.append(total)
                ax.plot(range(1, n + 1), cum, color=color, marker="o", markersize=2.7,
                        linewidth=1.15, zorder=3, label=label)
            ax.set_xticks(range(1, n + 1), [f"q{q}" for q in queries], rotation=90,
                          fontsize=7)
            ax.set_xlim(0.3, n + 0.7)
            ax.set_ylim(0, 35)
            ax.set_yticks(range(0, 36, 5))
            ax.set_xlabel("ClickBench query" if bench == "cb" else "TPC-H query", fontsize=10)
            if ax is axes[0]:
                ax.set_ylabel("Cumulative query time (s)", fontsize=10)
            ax.tick_params(axis="y", labelsize=9, length=2.5, width=0.5)
            ax.tick_params(axis="x", length=2.5, width=0.5, pad=2)
            ax.grid(axis="y", color="#eeeeee", linewidth=0.55)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#666666")
                spine.set_linewidth(0.55)
            gap = 0.06 * max(finals)
            placed = []
            for final in sorted(finals, reverse=True):
                y = final if not placed or placed[-1] - final >= gap else placed[-1] - gap
                placed.append(y)
                ax.annotate(f"{final:.1f}s", xy=(n, final), xytext=(1.018, y),
                            textcoords=ax.get_yaxis_transform(), ha="left", va="center",
                            fontsize=8.5, annotation_clip=False)
            ax.set_title(title, fontsize=11, fontweight="normal", pad=8)
            print(f"{title:11} totals " + "  ".join(f"{c[0]}={f:.2f}s" for c, f in zip(CUM_CONDS, finals)))
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, fontsize=8.8, frameon=True,
                            facecolor="white", edgecolor="#777777", fancybox=True,
                            framealpha=1, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.98),
                            handlelength=2.1, handletextpad=0.65, columnspacing=1.8,
                            borderpad=0.55, labelspacing=0.35)
        legend.get_frame().set_linewidth(0.55)
        fig.canvas.draw()
        widths = [ax.get_position().width for ax in axes]
        if abs(widths[0] - widths[1]) > 1e-12:
            raise ValueError(f"Panel widths differ: {widths}")
        print(f"Verified equal panel widths: {widths[0]:.6f} of figure width")
        for ext in ("png", "svg", "pdf"):
            path = OUT / f"cumulative_query_time_v3_styled.{ext}"
            fig.savefig(path, dpi=300, facecolor="white")
            print(path)
            matched_path = OUT / f"cumulative_query_time_v3_matched.{ext}"
            fig.savefig(matched_path, dpi=300, facecolor="white")
        import csv
        with (OUT / "cumulative_query_time_v3_styled.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(exported[0]))
            writer.writeheader()
            writer.writerows(exported)
        plt.close(fig)


def per_query_hits() -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [41, 26]})
    for ax, (title, bench, _, queries) in zip(axes, WORKLOADS):
        totals[title] = {}
        for label, cond, color, marker in CONDS:
            per, total = hit_by_query(bench, cond)
            totals[title][label] = total
            ax.plot(range(1, len(queries) + 1), [per.get(q, 0.0) for q in queries], color=color,
                    marker=marker, markersize=2.2, linewidth=1.0, zorder=3, label=label)
        style_axis(ax, [f"q{q}" for q in queries], "cache hit ratio (% of bytes)" if ax is axes[0] else None)
        ax.set_ylim(-3, 103)
        ax.set_title(title, fontsize=8.5, pad=4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, ncol=3, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "per_query_hit_v3")
    return totals


def summary(totals: dict[str, dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(3.1, 2.6))
    styles = {"TPC-H SF50": ("#b4522e", "o", "below"), "ClickBench": ("#3d6b66", "s", "above")}
    conds = ["paging", "paging + reorder"]
    for title, (color, marker, where) in styles.items():
        ys = [totals[title][c] for c in conds]
        ax.plot([0, 1], ys, color=color, marker=marker, markersize=4, linewidth=1.3, zorder=3,
                label=title)
        for x, y in zip((0, 1), ys):
            below = where == "below"
            ax.text(x, y - 2.2 if below else y + 2.0, f"{y:.1f}%", ha="center",
                    va="top" if below else "bottom", fontsize=7, color=color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(conds, fontsize=7.5)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(0, 50)
    ps.finish(ax, "cache hit ratio (% of bytes)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "cache_hit_summary_v3")


def save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300)
    print(OUT / f"{stem}.png")


def main() -> int:
    cumulative()
    summary(per_query_hits())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
