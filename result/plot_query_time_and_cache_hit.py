#!/usr/bin/env python3
"""Query wall-time curves plus paging/future-eviction byte hits on the right.

Full source includes all 43 ClickBench queries; reference source reproduces the
requested original A_* latency and legacy HIT_* plots with 41 ClickBench queries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_cumulative_scan_work import CONDS, ROOT, OUT
from analyze_full_query_latency_and_hits import hit_case_path
from plot_requested_cache_hits import collect, write_csv


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def prepare():
    query_path = OUT / "cumulative_query_time_full.csv"
    hit_path = OUT / "cache_hit_comparison_full.csv"
    query_rows, hit_rows = read_csv(query_path), read_csv(hit_path)
    assert len(query_rows) == 3 * (43 + 22) and len(hit_rows) == 4
    cases = []
    for title, bench, n in (("ClickBench", "cb", 43), ("TPC-H SF50", "th", 22)):
        for _, condition, _, _ in CONDS:
            subset = [r for r in query_rows if r["workload"] == title and r["condition"] == condition]
            assert len(subset) == n and {r["query"] for r in subset} == {f"q{q}" for q in range(1, n + 1)}
            case = ROOT / f"experiment/FULL_CACHE_PIN_20261007/runs/latency/{bench}_r1/{condition}"
            meta = json.loads((case / "metadata.json").read_text())
            assert meta["status"] == "complete" and meta["comparison_valid"], case
            assert meta["environment"]["SIRIUS_CACHE_HIT_TRACE"] == "0", case
            records = read_csv(case / "queries.csv")
            assert len(records) == 3 * n and all(r["status"] == "ok" for r in records), case
            cumulative = 0.0
            for q, row in enumerate(subset, 1):
                assert row["query"] == f"q{q}" and int(row["samples"]) == 2
                values = [float(r["wall_s"]) for r in records if int(r["query"]) == q
                          and int(r["execution"]) in (2, 3)]
                assert len(values) == 2
                mean = statistics.mean(values)
                cumulative += mean
                assert math.isclose(mean, float(row["query_wall_s"]), abs_tol=1e-8)
                assert math.isclose(cumulative, float(row["cumulative_query_wall_s"]), abs_tol=1e-8)
            assert math.isclose(cumulative, meta["steady_state_workload_wall_s"], abs_tol=1e-8)
            cases.append(dict(path=str(case), metadata=meta))
            if condition != "baseline":
                hit_case = hit_case_path(bench, condition)
                hit_meta = json.loads((hit_case / "metadata.json").read_text())
                assert hit_meta["status"] == "complete" and hit_meta["comparison_valid"], hit_case
                assert hit_meta["environment"]["SIRIUS_CACHE_HIT_TRACE"] == "1", hit_case
                for field in ("order", "executions", "gpu_device", "cache_budget_bytes",
                              "min_free_bytes_per_gpu", "future_mode", "future_horizon",
                              "late_materialization", "prefetch", "odirect"):
                    assert hit_meta[field] == meta[field], (title, field)
                hit = [r for r in hit_rows if r["workload"] == title and r["condition"] == condition]
                assert len(hit) == 1
                hit = hit[0]
                assert int(hit["query_count"]) == n and hit["metric"] == "value_bytes_v2"
                total = int(hit["total_decoded_value_bytes"])
                cached = int(hit["cache_served_decoded_value_bytes"])
                assert total > 0 and total - int(hit["reader_decoded_value_bytes"]) == cached
                assert math.isclose(100 * cached / total, float(hit["cache_hit_pct"]), abs_tol=1e-8)
                cases.append(dict(path=str(hit_case), metadata=hit_meta))
    return query_rows, hit_rows, dict(
        cases=cases, warmup_execution=1, measured_executions=[2, 3], independent_runs=1,
        query_metric="Mean per-query execute().fetchall() elapsed time, accumulated in query-number order",
        hit_metric="100 * total cache-served decoded value bytes / total traced decoded value bytes; repeated accesses included",
        timing_and_hit_runs="Separate latency and cache-hit diagnostic cases; configuration matches within each workload/condition. TPC-H paging uses a separately completed same-configuration retry.",
        source_files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                      for p in (query_path, hit_path)])


def prepare_reference():
    query_path = OUT / "cumulative_query_time_v3_styled.csv"
    original = read_csv(query_path)
    assert len(original) == 3 * (41 + 22)
    query_rows, cases = [], []
    done = re.compile(r"exec=(\d+)/\d+ .* q(\d+) done wall=([\d.]+)s")
    for title, bench, reps, queries in (
        ("ClickBench", "cb", (1, 2), [q for q in range(1, 44) if q not in (28, 29)]),
        ("TPC-H SF50", "th", (1, 2, 3), list(range(1, 23))),
    ):
        for label, condition, old_condition, _ in CONDS:
            sample_values = {}
            for run in reps:
                log = ROOT / f"experiment/A_{bench}_{old_condition}_r{run}/runner.log"
                records = [m.groups() for line in log.read_text().splitlines() if (m := done.search(line))]
                expected = {(q, ex) for q in queries for ex in (1, 2, 3)}
                assert len(records) == len(expected), log
                assert {(int(q), int(ex)) for ex, q, wall in records} == expected, log
                for ex, q, wall in records:
                    if int(ex) in (2, 3):
                        sample_values.setdefault(int(q), []).append(float(wall))
                cases.append(dict(path=str(log), sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
            total = 0.0
            for position, q in enumerate(queries, 1):
                mean = statistics.mean(sample_values[q])
                total += mean
                stored = [r for r in original if r["workload"] == title and r["condition"] == label and r["query"] == f"q{q}"]
                assert len(stored) == 1
                assert math.isclose(mean, float(stored[0]["wall_s"]), abs_tol=1e-8)
                assert math.isclose(total, float(stored[0]["cumulative_wall_s"]), abs_tol=1e-8)
                query_rows.append(dict(workload=title, condition=condition, query=f"q{q}", position=position,
                                       query_wall_s=mean, cumulative_query_wall_s=total))
            print(f"Reference {title} {condition}: {total:.6f}s ({len(queries)} queries)", flush=True)
    summary, _, _, _, audit = collect("legacy", ROOT / "experiment/FULL_CACHE_PIN_20261007/runs")
    hit_rows = [dict(workload="ClickBench" if r["workload"] == "cb" else "TPC-H SF50",
                     condition=r["condition"], query_count=r["query_count"], cache_hit_pct=r["byte_hit_pct"],
                     total_bytes=r["total_bytes"], reader_bytes=r["reader_bytes"], cached_bytes=r["cached_bytes"],
                     metric=r["metric"]) for r in summary]
    write_csv(OUT / "query_latency_reference_41.csv", query_rows)
    write_csv(OUT / "cache_hit_reference_41.csv", hit_rows)
    return query_rows, hit_rows, dict(
        source="requested original reference plots", latency_cases=cases, hit_cases=audit,
        query_counts={"ClickBench": 41, "TPC-H SF50": 22}, excluded_clickbench_queries=[28, 29],
        warmup_execution=1, measured_executions=[2, 3],
        independent_latency_runs={"ClickBench": 2, "TPC-H SF50": 3}, independent_hit_runs=1,
        query_metric="Original A_* per-query execute().fetchall() wall time, accumulated in query-number order",
        hit_metric="Original legacy Parquet-scan traced-byte ratio, not value_bytes_v2 or conditional 100%",
        source_files=[dict(path=str(query_path), sha256=hashlib.sha256(query_path.read_bytes()).hexdigest())])


def finish(ax):
    ax.grid(axis="y", color="#eeeeee", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2.5, width=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.55)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("full", "reference", "illustrative"), default="full")
    args = parser.parse_args()
    illustrative = args.source == "illustrative"
    reference = args.source != "full"
    query_rows, hit_rows, provenance = prepare_reference() if reference else prepare()
    if illustrative:
        augmented = []
        for title in ("ClickBench", "TPC-H SF50"):
            for _, condition, _, _ in CONDS:
                originals = {r["query"]: r for r in query_rows
                             if r["workload"] == title and r["condition"] == condition}
                total = 0.0
                for q in range(1, 44 if title == "ClickBench" else 23):
                    assumed = title == "ClickBench" and q in (28, 29)
                    wall = 0.1 if assumed else originals[f"q{q}"]["query_wall_s"]
                    total += wall
                    augmented.append(dict(workload=title, condition=condition, query=f"q{q}",
                                          position=q, query_wall_s=wall, cumulative_query_wall_s=total,
                                          value_type="assumed_example" if assumed else "measured_reference"))
        query_rows = augmented
        provenance.update(source="explicitly illustrative draft, not a measured 43-query result",
                          query_metric="Measured reference query wall times plus assumed q28/q29 times",
                          assumed_clickbench_query_times_s={"q28": 0.1, "q29": 0.1},
                          assumed_conditions=[c[1] for c in CONDS],
                          displayed_query_counts={"ClickBench": 43, "TPC-H SF50": 22},
                          cache_hit_scope="Measured legacy 41-query ClickBench / 22-query TPC-H traces")
        write_csv(OUT / "query_latency_illustrative_q28_q29.csv", query_rows)
    prefix = OUT / ("query_latency_and_cache_hit_illustrative_q28_q29" if illustrative else
                    "query_latency_and_cache_hit_reference_41" if reference else
                    "query_latency_and_cache_hit_comparison_full")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.bbox": None}):
        width, height = 14.5, 3.9
        fig = plt.figure(figsize=(width, height))
        # Preserve the previous time panels' 4.06 x 2.301 inch inner frames.
        axes = [fig.add_axes([x / width, 0.78 / height, w / width, 2.301 / height])
                for x, w in ((0.70, 4.06), (5.72, 4.06), (11.05, 2.95))]
        for ax, title, n in zip(axes[:2], ("ClickBench", "TPC-H SF50"),
                               (41 if args.source == "reference" else 43, 22)):
            finals = []
            positions = [i / (n - 1) for i in range(n)]
            for label, condition, _, color in CONDS:
                rows = [r for r in query_rows if r["workload"] == title and r["condition"] == condition]
                ys = [float(r["cumulative_query_wall_s"]) for r in rows]
                ax.plot(positions, ys, color=color, marker="o", markersize=2.7,
                        linewidth=1.15, zorder=3, label=label)
                finals.append(ys[-1])
            tick_labels = [r["query"] + ("*" if illustrative and title == "ClickBench"
                                          and r["query"] in ("q28", "q29") else "") for r in rows]
            ax.set_xticks(positions, tick_labels, rotation=90, fontsize=7)
            ax.tick_params(axis="x", pad=2)
            ax.set_xlim(-0.025, 1.025)
            ymax, step = (100, 10) if title == "ClickBench" and not reference else (35, 5)
            ax.set_ylim(0, ymax)
            ax.set_yticks(range(0, ymax + 1, step))
            ax.set_xlabel("ClickBench query" if title == "ClickBench" else "TPC-H query", fontsize=10)
            if title == "ClickBench":
                ax.set_ylabel("Cumulative query latency (s)", fontsize=10)
            panel_title = ("ClickBench (illustrative q28/q29)" if illustrative and title == "ClickBench" else
                           f"{title} ({n} queries)" if reference else title)
            ax.set_title(panel_title,
                         fontsize=11, fontweight="normal", pad=8)
            if illustrative and title == "ClickBench":
                ax.text(0.03, 0.965, "ILLUSTRATIVE DRAFT", transform=ax.transAxes,
                        color="#a34535", fontsize=8.5, fontweight="bold", va="top")
                ax.text(0.03, 0.905, "* q28/q29: assumed 0.1s each, not measured", transform=ax.transAxes,
                        color="#a34535", fontsize=8, va="top")
            last_y = None
            for final in sorted(finals, reverse=True):
                y = final if last_y is None else min(final, last_y - 0.04 * ymax)
                last_y = y
                ax.annotate(f"{final:.1f}s", xy=(1, final), xytext=(1.018, y),
                            textcoords=ax.get_yaxis_transform(), ha="left", va="center",
                            fontsize=8.5, annotation_clip=False)
            finish(ax)

        ax = axes[2]
        for title, color, marker, dy in (("TPC-H SF50", "#803e26", "o", -9),
                                        ("ClickBench", "#3d6b66", "s", 9)):
            values = [float(next(r for r in hit_rows if r["workload"] == title and r["condition"] == condition)["cache_hit_pct"])
                      for condition in ("paging_lru", "reorder_future")]
            ax.plot([0, 1], values, color=color, marker=marker, linewidth=1.6, markersize=5.2,
                    label=title, zorder=3)
            for x, value in enumerate(values):
                ax.annotate(f"{value:.1f}%", xy=(x, value), xytext=(0, dy), textcoords="offset points",
                            ha="center", va="bottom" if dy > 0 else "top", fontsize=10,
                            fontstyle="italic", color=color)
        ax.set_xticks([0, 1], ["paging", "paging + reorder\n+ future eviction"], fontsize=8.5)
        ax.tick_params(axis="x", length=0, pad=7)
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(0, 32 if reference else 50)
        ax.set_yticks(range(0, 31, 5) if reference else range(0, 51, 10))
        ax.set_ylabel("Cache hit ratio (% of traced bytes)" if reference else
                      "Cache hit ratio (% of bytes)", fontsize=9.5)
        ax.set_title("Cache hit ratio", fontsize=11, fontweight="normal", pad=8)
        finish(ax)
        ax.tick_params(axis="x", length=2.5)
        ax.spines[["top", "right"]].set_visible(False)
        hit_handles, hit_labels = ax.get_legend_handles_labels()
        fig.legend(hit_handles, hit_labels, fontsize=9, frameon=False, ncol=1,
                   loc="upper center", bbox_to_anchor=(12.525 / width, 0.98),
                   handlelength=2.1, labelspacing=0.35)

        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, fontsize=8.8, frameon=True,
                            facecolor="white", edgecolor="#777777", fancybox=True,
                            framealpha=1, ncol=3, loc="upper center",
                            bbox_to_anchor=(5.24 / width, 0.98), handlelength=2.1,
                            handletextpad=0.65, columnspacing=1.8, borderpad=0.55)
        legend.get_frame().set_linewidth(0.55)
        footnote = ("ILLUSTRATIVE: ClickBench q28/q29 assumed 0.1s each. "
                    "Cache hits measured on 41 ClickBench / 22 TPC-H queries; legacy byte metric."
                    if illustrative else
                    "ClickBench: 41 queries (q28/q29 excluded). TPC-H: 22 queries. "
                    "Legacy Parquet-scan byte metric; warmup excluded." if reference else
                    "Cache hit ratio = cache-served decoded value bytes / total traced decoded value bytes. "
                    "All queries; warmup excluded.")
        fig.text(0.5, 0.025, footnote, ha="center", fontsize=7.5,
                 color="#a34535" if illustrative else "#666666")
        fig.canvas.draw()
        bounds = [list(ax.get_position().bounds) for ax in axes]
        assert abs(bounds[0][2] - bounds[1][2]) < 1e-12
        spans = [float(ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0])
                 * 300 / fig.dpi for ax in axes[:2]]
        assert abs(spans[0] - spans[1]) < 1e-8
        provenance["layout"] = dict(figsize_inches=[width, height], axes_bounds=bounds,
            time_panel_width_pixels=[b[2] * width * 300 for b in bounds[:2]],
            time_panel_data_width_pixels=spans)
        Path(str(prefix) + "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        for ext in ("png", "svg", "pdf"):
            path = prefix.with_suffix(f".{ext}")
            fig.savefig(path, dpi=300, facecolor="white")
            print(path, flush=True)
        plt.close(fig)
    print(f"Verified time-panel frame widths: {provenance['layout']['time_panel_width_pixels']}")
    print(f"Verified time-panel data widths: {spans}")


if __name__ == "__main__":
    main()
