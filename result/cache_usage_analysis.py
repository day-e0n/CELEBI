#!/usr/bin/env python3
"""Offline cache-use counts and logical-cell coverage for the exact headline FA runs.

Run: pixi run python result/cache_usage_analysis.py

No benchmark is executed. Only base/lru/ro_lru, executions 2–3, are included.
Each materialize audit is a batch. Full-hit audits and successful partial_serve
records count actual reuse; column_splice cached>0/read>0 counts partial-column
reuse for these non-partitioned benchmark files. Other batches are called
'no observed reuse'. Partial-read cells are row*column cells, NOT bytes.

For full hits, cached cells = audit output_rows*output_columns. For successful
partial serving, cached cells = total_cells-read_cells. For column splicing,
markers contain only file and row-group COUNT, without task/thread identity.
Candidate audit rows are grouped by query/execution/file/RG count/column count.
All possible assignments give the conservative minimum and maximum cached-cell
counts. Identical candidate row counts make this exact; otherwise no assignment
is guessed. Denominator uses cache-condition output before post-filtering.
Baseline audit output cells differ because its reader can filter sooner.
"""
from __future__ import annotations

import collections
import csv
import importlib.util
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
LABEL = re.compile(r"sirius_set_query_label\('\w+_q(\d+)_exec(\d+)'\)")
FIELDS = re.compile(r"\b(\w+)=(\d+)\b")
WORKLOADS = (("ClickBench", "cb", (1, 2), tuple(q for q in range(1, 44) if q not in (28, 29))),
             ("TPC-H SF50", "th", (1, 2, 3), tuple(range(1, 23))))
CONDITIONS = ("base", "lru", "ro_lru")


def parse_run(run: Path):
    by_query = collections.defaultdict(lambda: {"audits": [], "partial": [], "splice": []})
    key = None
    for line in next(run.glob("*/log_dir/*.log")).open():
        if m := LABEL.search(line):
            key = tuple(map(int, m.groups()))
            continue
        if key is None or key[1] not in (2, 3):
            continue
        f = {k: int(v) for k, v in FIELDS.findall(line)}
        if "[scan-audit] parquet_materialize" in line:
            file = re.search(r" files=(\S+)", line)[1]
            groups = re.search(r" row_groups=(\S+)", line)[1]
            if "," in file or f["split_count"] != 1:
                raise ValueError(f"Unsupported multi-file audit: {line}")
            f["bin"] = file, len(groups.split(",")), f["output_columns"]
            by_query[key]["audits"].append(f)
        elif "[fixed-page-cache] partial_serve" in line:
            by_query[key]["partial"].append(f)
        elif "[fixed-page-cache] column_splice" in line and f["cached"] > 0 and f["read"] > 0:
            file = re.search(r" file='([^']+)'", line)[1]
            f["bin"] = file, f["row_groups"], f["cached"] + f["read"]
            by_query[key]["splice"].append(f)
    return by_query


def measure(data):
    audits, partials, splices = data["audits"], data["partial"], data["splice"]
    full = [a for a in audits if a["full_hit"]]
    partial = [p for p in partials if p["used"]]
    # These saved logs have zero failed partial layouts; preserve an explicit
    # check because ps_read_us may otherwise describe an abandoned attempt.
    failed = sum(not p["used"] for p in partials)
    if failed:
        raise ValueError("Failed partial layout makes the audit classification ambiguous")
    pa = [a for a in audits if not a["full_hit"] and a.get("ps_read_us", 0) > 0]
    if len(pa) != len(partial):
        raise ValueError("Successful partial markers do not match partial-read audits")
    if sum(a["output_rows"] * a["output_columns"] for a in pa) != sum(p["total_cells"] for p in partial):
        raise ValueError("Partial total_cells differs from audited output cell counts")
    bins = collections.defaultdict(list)
    weights = collections.defaultdict(list)
    for a in audits:
        if not a["full_hit"] and not a.get("ps_read_us", 0):
            bins[a["bin"]].append(a["output_rows"])
    for s in splices:
        weights[s["bin"]].append(s["cached"])
    lo, hi, ambiguous = 0, 0, 0
    for key, cache_cols in weights.items():
        row_counts = bins[key]
        if len(cache_cols) > len(row_counts):
            raise ValueError(f"Splice count exceeds candidate audits for {key}")
        # Include zero weights for candidates without a splice marker. The
        # rearrangement inequality bounds every consistent possible assignment.
        ws = sorted(cache_cols + [0] * (len(row_counts) - len(cache_cols)), reverse=True)
        lower = sum(r * w for r, w in zip(sorted(row_counts), ws))
        upper = sum(r * w for r, w in zip(sorted(row_counts, reverse=True), ws))
        lo += lower
        hi += upper
        ambiguous += lower != upper
    full_cells = sum(a["output_rows"] * a["output_columns"] for a in full)
    partial_total = sum(p["total_cells"] for p in partial)
    partial_cached = sum(p["total_cells"] - p["read_cells"] for p in partial)
    for p in partial:
        if not 0 < p["total_cells"] - p["read_cells"] < p["total_cells"]:
            raise ValueError("Partial serving did not have a strict subset of cached cells")
    reuse = len(full) + len(partial) + len(splices)
    if reuse > len(audits):
        raise ValueError("Reuse classifications overlap or exceed audit count")
    return collections.Counter({
        "audit_batches": len(audits), "full_hit_batches": len(full),
        "partial_page_batches": len(partial), "partial_column_batches": len(splices),
        "no_observed_reuse_batches": len(audits) - reuse,
        "zero_output_rows_batches": sum(a["output_rows"] == 0 for a in audits),
        "partial_failed_batches": failed, "ambiguous_column_splice_bins": ambiguous,
        "audited_output_cells": sum(a["output_rows"] * a["output_columns"] for a in audits),
        "full_cached_cells": full_cells, "partial_page_total_cells": partial_total,
        "partial_page_cached_cells": partial_cached,
        "partial_column_cached_cells_lower": lo, "partial_column_cached_cells_upper": hi,
        "cached_cells_lower": full_cells + partial_cached + lo,
        "cached_cells_upper": full_cells + partial_cached + hi})


def format_row(title, condition, query, metrics, samples):
    row = {"workload": title, "condition": condition, "query": query, "samples": samples}
    row.update({k: v / samples for k, v in metrics.items()})
    n = metrics["audit_batches"]
    for k in ("full_hit_batches", "partial_page_batches", "partial_column_batches", "no_observed_reuse_batches"):
        row[k + "_fraction"] = metrics[k] / n if n else 0
    row["any_reuse_batch_fraction"] = (n - metrics["no_observed_reuse_batches"]) / n if n else 0
    cells = metrics["audited_output_cells"]
    for suffix in ("lower", "upper"):
        row[f"cached_logical_cell_fraction_{suffix}"] = metrics[f"cached_cells_{suffix}"] / cells if cells else 0
    pcells = metrics["partial_page_total_cells"]
    row["partial_page_cached_cell_fraction_within_partial_only"] = metrics["partial_page_cached_cells"] / pcells if pcells else 0
    return row


def main():
    # Reuse only the completeness validator; importing does not generate artifacts.
    spec = importlib.util.spec_from_file_location("headline", OUT / "analyze_headline_three_conditions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows, totals = [], []
    for title, bench, reps, queries in WORKLOADS:
        for condition in CONDITIONS:
            total = collections.Counter()
            qs = collections.defaultdict(collections.Counter)
            for r in reps:
                run = ROOT / "experiment" / f"FA_{bench}_{condition}_r{r}"
                module.parse_wall(run, queries)
                by_query = parse_run(run)
                for q in queries:
                    for e in (2, 3):
                        m = measure(by_query[q, e])
                        qs[q].update(m)
                        total.update(m)
            samples = 2 * len(reps)
            for q in queries:
                rows.append(format_row(title, condition, f"q{q}", qs[q], samples))
            totals.append(format_row(title, condition, "total", total, samples))
    for name, output in (("cache_usage_queries.csv", rows), ("cache_usage_summary.csv", totals)):
        with (OUT / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), constrained_layout=True)
    parts = (("full_hit_batches_fraction", "Full batch", "#3d6b66"),
             ("partial_page_batches_fraction", "Partial pages", "#75a8a2"),
             ("partial_column_batches_fraction", "Partial columns", "#b4522e"),
             ("no_observed_reuse_batches_fraction", "No observed reuse", "#d5dad8"))
    for ax, (title, *_rest) in zip(axes, WORKLOADS):
        selected = [r for r in totals if r["workload"] == title]
        for x, r in enumerate(selected):
            bottom = 0
            for key, label, color in parts:
                val = r[key] * 100
                ax.bar(x, val, bottom=bottom, color=color, edgecolor="white", linewidth=0.7,
                       label=label if x == 0 else None, width=0.6)
                if val >= 4:
                    ax.text(x, bottom + val / 2, f"{val:.1f}%", ha="center", va="center", fontsize=8)
                bottom += val
        ax.set_xticks(range(3), ["baseline", "paging", "paging\n+ reorder"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Share of scan batches (%)")
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Observed GPU page-cache reuse: exact headline runs, executions 2–3\n"
                 "Batch-count shares; these percentages are not byte shares.", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"cache_usage_batch_shares.{ext}", dpi=220)
    print(json.dumps(totals, indent=2))
    print("Validated complete queries; partial markers/cells agree with audits; failed partials checked.")


if __name__ == "__main__":
    main()
