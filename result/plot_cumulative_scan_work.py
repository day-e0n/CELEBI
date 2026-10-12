#!/usr/bin/env python3
"""Plot scan Computing dwell time from completed latency telemetry.

Full: all ClickBench 43 / TPC-H 22 queries, one run, executions 2-3.
Reference: the original A_* figure's 41 / 22 queries, two / three runs.
Both accumulate per-query means in query-number order, not arrival order.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result"
sys.path.insert(0, str(ROOT / "scripts"))
from parse_quent_operator_breakdown import (  # noqa: E402
    accumulate_task_durations, bucket_for, load_telemetry, parse_label,
)
import matplotlib.pyplot as plt  # noqa: E402

CONDS = [
    ("Baseline", "baseline", "base", "#4c78a8"),
    ("Baseline+page caching", "paging_lru", "lru", "#59a14f"),
    ("Baseline+page caching+reordering (CELEBI)", "reorder_future", "best", "#f2b134"),
]


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract(source):
    samples, raw, provenance, means = [], [], [], []
    for title, bench, last_query in [("ClickBench", "cb", 43), ("TPC-H SF50", "th", 22)]:
        queries = [q for q in range(1, last_query + 1)
                   if source == "full" or bench != "cb" or q not in (28, 29)]
        reps = (1,) if source == "full" else ((1, 2) if bench == "cb" else (1, 2, 3))
        for label, condition, old_condition, _ in CONDS:
            for run in reps:
                if source == "full":
                    case = ROOT / f"experiment/FULL_CACHE_PIN_20261007/runs/latency/{bench}_r{run}/{condition}"
                    metadata = json.loads((case / "metadata.json").read_text())
                    assert metadata["status"] == "complete" and metadata["comparison_valid"], case
                    assert metadata["environment"]["SIRIUS_CACHE_HIT_TRACE"] == "0", case
                    assert metadata["future_mode"] == ("evict" if condition == "reorder_future" else "off"), case
                    assert metadata["future_horizon"] == "execution", case
                    with (case / "queries.csv").open() as f:
                        records = list(csv.DictReader(f))
                    expected = {(q, ex) for q in queries for ex in (1, 2, 3)}
                    assert len(records) == len(expected), case
                    assert {(int(r["query"]), int(r["execution"])) for r in records} == expected, case
                    assert all(r["status"] == "ok" for r in records), case
                    query_labels = {(int(r["query"]), int(r["execution"])): r["label"] for r in records}
                else:
                    sub = "baseline" if condition == "baseline" else "celebi_fixed_variable"
                    case = ROOT / f"experiment/A_{bench}_{old_condition}_r{run}/{sub}"
                    metadata = None
                    query_labels = {}
                telemetry = case / "telemetry_data"
                mappings = load_telemetry(telemetry)
                totals = accumulate_task_durations(telemetry, mappings[1], mappings[2], mappings[3], mappings[0])
                if source == "reference":
                    for name in totals:
                        parsed = parse_label(name)
                        if parsed and parsed[1] in queries and parsed[2] in (1, 2, 3):
                            key = parsed[1:]
                            assert key not in query_labels, (case, key)
                            query_labels[key] = name
                    assert set(query_labels) == {(q, ex) for q in queries for ex in (1, 2, 3)}, case
                hint_count = 0
                if condition == "reorder_future":
                    for log in (case / "log_dir").glob("*.log"):
                        for line in log.read_text().splitlines():
                            if "QueryBegin: SQL: CALL sirius_set_cache_hints('" in line:
                                assert "sirius_set_cache_hints('evict;" in line, (case, line)
                                hint_count += 1
                    assert hint_count == len(queries) * 3, (case, hint_count)
                for q in queries:
                    for execution in (2, 3):
                        name = query_labels[q, execution]
                        assert name in totals, (case, name)
                        kinds = totals[name]
                        assert kinds.get("UNKNOWN", 0) == 0, (case, name)
                        scan_ms = sum(ms for kind, ms in kinds.items() if bucket_for(kind) == "scan")
                        assert scan_ms > 0, (case, name)
                        sample = dict(workload=title, condition=condition, query=f"q{q}",
                                      run=run, execution=execution, scan_work_s=scan_ms / 1000)
                        samples.append(sample)
                        for kind, ms in kinds.items():
                            if bucket_for(kind) == "scan":
                                raw.append(dict(workload=title, condition=condition, query=f"q{q}",
                                                run=run, execution=execution, operator_kind=kind,
                                                scan_work_s=ms / 1000))
                provenance.append(dict(workload=title, condition=condition, run=run, case=str(case),
                    metadata=metadata, future_eviction_calls=hint_count,
                    telemetry_files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                     for p in sorted(telemetry.rglob("*.ndjson"))]))
            cumulative = 0.0
            for position, q in enumerate(queries, 1):
                values = [r["scan_work_s"] for r in samples if r["workload"] == title
                          and r["condition"] == condition and r["query"] == f"q{q}"]
                assert len(values) == 2 * len(reps), (title, condition, q)
                mean = statistics.mean(values)
                cumulative += mean
                means.append(dict(workload=title, condition=condition, label=label, query=f"q{q}",
                                  position=position, samples=len(values), scan_work_s=mean,
                                  scan_work_sd_s=statistics.stdev(values), cumulative_scan_work_s=cumulative))
            print(f"{source}: {title} {condition}: {cumulative:.6f}s ({len(queries)} queries)", flush=True)
    prefix = OUT / f"cumulative_scan_work_{source}"
    write_csv(prefix.with_suffix(".csv"), means)
    write_csv(Path(str(prefix) + "_samples.csv"), samples)
    write_csv(Path(str(prefix) + "_operator_kinds.csv"), raw)
    Path(str(prefix) + "_provenance.json").write_text(json.dumps(dict(
        source=source, definition="Sum of scan-attributed Task Computing dwell times. Includes host work and waits; overlapping tasks add. Not query wall time or GPU kernel active time.",
        accumulation="Per-query means accumulated in query-number order, not actual execution order.",
        warmup_execution=1, averaged_executions=[2, 3], cases=provenance), indent=2) + "\n")
    return means, prefix


def plot(means, prefix, value_key="cumulative_scan_work_s",
         ylabel="Cumulative scan work (s)", tick_step=None, label_gap_fraction=0.065,
         equal_data_span=False):
    maximum = max(r[value_key] for r in means)
    step = tick_step or (25 if maximum < 200 else 50 if maximum < 400 else 100)
    ymax = math.ceil(maximum * 1.07 / step) * step
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.linewidth": 0.55, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "savefig.bbox": None}):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1, 1]})
        fig.subplots_adjust(left=0.065, right=0.935, bottom=0.20, top=0.79, wspace=0.25)
        for ax, title in zip(axes, ("ClickBench", "TPC-H SF50")):
            finals = []
            for label, condition, _, color in CONDS:
                rows = [r for r in means if r["workload"] == title and r["condition"] == condition]
                positions = ([(r["position"] - 1) / (len(rows) - 1) for r in rows]
                             if equal_data_span else [r["position"] for r in rows])
                ax.plot(positions, [r[value_key] for r in rows],
                        color=color, marker="o", markersize=2.7, linewidth=1.15, zorder=3, label=label)
                finals.append(rows[-1][value_key])
            n = len(rows)
            ax.set_xticks(positions, [r["query"] for r in rows], rotation=90, fontsize=7)
            ax.set_xlim(-0.025, 1.025) if equal_data_span else ax.set_xlim(0.3, n + 0.7)
            ax.set_ylim(0, ymax)
            ax.set_yticks(range(0, ymax + 1, step))
            ax.set_xlabel("ClickBench query" if title == "ClickBench" else "TPC-H query", fontsize=10)
            if ax is axes[0]:
                ax.set_ylabel(ylabel, fontsize=10)
            ax.tick_params(axis="y", labelsize=9, length=2.5, width=0.5)
            ax.tick_params(axis="x", length=2.5, width=0.5, pad=2)
            ax.grid(axis="y", color="#eeeeee", linewidth=0.55)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#666666")
                spine.set_linewidth(0.55)
            last_y = None
            for final in sorted(finals, reverse=True):
                y = final if last_y is None else min(final, last_y - label_gap_fraction * ymax)
                last_y = y
                ax.annotate(f"{final:.1f}s", xy=(positions[-1], final), xytext=(1.018, y),
                            textcoords=ax.get_yaxis_transform(), ha="left", va="center",
                            fontsize=8.5, annotation_clip=False)
            ax.set_title(title, fontsize=11, fontweight="normal", pad=8)
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, fontsize=8.8, frameon=True, facecolor="white",
                            edgecolor="#777777", fancybox=True, framealpha=1, ncol=3,
                            loc="upper center", bbox_to_anchor=(0.5, 0.98), handlelength=2.1,
                            handletextpad=0.65, columnspacing=1.8, borderpad=0.55, labelspacing=0.35)
        legend.get_frame().set_linewidth(0.55)
        fig.canvas.draw()
        bounds = [list(ax.get_position().bounds) for ax in axes]
        assert abs(bounds[0][2] - bounds[1][2]) < 1e-12
        assert abs(bounds[0][3] - bounds[1][3]) < 1e-12
        data_spans = []
        for ax in axes:
            line = ax.lines[0]
            xs = line.get_xdata()
            data_spans.append([float(ax.transData.transform((x, 0))[0]) * 300 / fig.dpi
                               for x in (xs[0], xs[-1])])
        if equal_data_span:
            assert abs((data_spans[0][1] - data_spans[0][0]) -
                       (data_spans[1][1] - data_spans[1][0])) < 1e-8
        Path(str(prefix) + "_layout.json").write_text(json.dumps(dict(
            figsize_inches=[10.5, 3.9], dpi=300, axes_bounds=bounds,
            axes_pixels=[dict(width=b[2] * 3150, height=b[3] * 1170) for b in bounds],
            common_ylim=[0, ymax], equal_data_span=equal_data_span,
            first_last_query_pixels=data_spans,
            plotted_data_width_pixels=[p[1] - p[0] for p in data_spans]), indent=2) + "\n")
        for ext in ("png", "svg", "pdf"):
            path = prefix.with_suffix(f".{ext}")
            fig.savefig(path, dpi=300, facecolor="white")
            print(path, flush=True)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("full", "reference", "both"), default="both")
    args = parser.parse_args()
    for source in (("full", "reference") if args.source == "both" else (args.source,)):
        plot(*extract(source))
