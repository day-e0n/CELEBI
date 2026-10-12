#!/usr/bin/env python3
"""Audited paging/future-eviction hit plots, from legacy or full-workload traces."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
LABEL = re.compile(r"sirius_set_query_label\('([^']*)'\)")
QUERY = re.compile(r"(.+)_q(\d+)_exec(\d+)$")
FIELDS = re.compile(r"(\w+)=('(?:[^']*)'|\S+)")
DONE = re.compile(r"exec=(\d+)/3 pos=\d+/\d+ q(\d+) done wall=")
CONDS = ("paging_lru", "reorder_future")
TITLES = {"th": "TPC-H SF50", "cb": "ClickBench"}
COLORS = {"th": "#803e26", "cb": "#3d6b66"}
KINDS = {"full", "splice", "partial", "miss"}


class NotReady(Exception):
    pass


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentage(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def parse_logs(logs, queries, expected_label_condition, metric):
    data = collections.defaultdict(collections.Counter)
    excluded = collections.Counter()
    for log in logs:
        active = None
        for line in log.open(errors="replace"):
            label = LABEL.search(line)
            if label:
                match = QUERY.fullmatch(label[1])
                active = (match[1], int(match[2]), int(match[3])) if match else None
                continue
            if not active or active[2] not in (2, 3) or "[cache-hit]" not in line:
                continue
            fields = {key: value.strip("'") for key, value in FIELDS.findall(line)}
            actual_metric = fields.get("metric", "legacy_parquet_bytes")
            if actual_metric != metric:
                excluded[actual_metric] += 1
                continue
            if active[0] != expected_label_condition or active[1] not in queries:
                raise ValueError(f"Wrong trace label: {log}, {active}")
            total, read = int(fields["total_bytes"]), int(fields["read_bytes"])
            if total < 0 or not 0 <= read <= total or fields["kind"] not in KINDS:
                raise ValueError(f"Invalid trace: {log}, {line.strip()}")
            path = fields.get("path", "parquet")
            if metric == "value_bytes_v2" and path not in {"parquet", "cached_provider", "late_refetch"}:
                raise ValueError(f"Unknown metric path: {log}, {path}")
            counter = data[active[1], active[2], path]
            counter["total_bytes"] += total
            counter["read_bytes"] += read
            counter["trace_batches"] += 1
            if total:
                counter["payload_batches"] += 1
                counter["full_batches"] += fields["kind"] == "full"
    traced = {(q, ex) for q, ex, _ in data}
    expected = {(q, ex) for q in queries for ex in (2, 3)}
    if traced != expected:
        raise ValueError(f"Missing/extra query traces: missing={sorted(expected-traced)}, extra={sorted(traced-expected)}")
    return data, excluded


def collect(source, input_root):
    cases, pending = [], []
    for workload in ("th", "cb"):
        queries = set(range(1, 23 if workload == "th" else 44))
        if source == "legacy" and workload == "cb":
            queries -= {28, 29}
        for condition in CONDS:
            metadata = None
            if source == "legacy":
                suffix = "lru" if condition == "paging_lru" else "ro_evict"
                case = ROOT / f"experiment/HIT_{workload}_{suffix}"
                runner = case / "runner.log"
                done = [tuple(map(int, m.groups())) for line in runner.read_text().splitlines()
                        if (m := DONE.search(line))]
                expected = {(ex, q) for ex in (1, 2, 3) for q in queries}
                if len(done) != len(expected) or set(done) != expected:
                    raise ValueError(f"Incomplete/duplicate legacy query execution: {case}")
                logs = sorted(case.glob("*/log_dir/*.log"))
                label_condition, metric = "paging", "legacy_parquet_bytes"
            else:
                case = input_root / f"hit/{workload}_r1" / condition
                metadata_path = case / "metadata.json"
                if not metadata_path.exists():
                    pending.append(f"{workload}/{condition}: missing metadata")
                    continue
                metadata = json.loads(metadata_path.read_text())
                status = metadata.get("status")
                if status in {"prepared", "running"}:
                    pending.append(f"{workload}/{condition}: {status}")
                    continue
                if status != "complete" or metadata.get("comparison_valid") is not True:
                    raise ValueError(f"Failed comparison: {case}, status={status}")
                if (metadata["environment"]["SIRIUS_CACHE_HIT_TRACE"] != "1"
                        or metadata["future_mode"] != ("off" if condition == "paging_lru" else "evict")
                        or metadata["future_horizon"] != "execution"):
                    raise ValueError(f"Wrong cache/future trace condition: {case}")
                with (case / "queries.csv").open(newline="") as handle:
                    records = list(csv.DictReader(handle))
                expected = {(ex, q) for ex in (1, 2, 3) for q in queries}
                if (len(records) != len(expected)
                        or {(int(r["execution"]), int(r["query"])) for r in records} != expected
                        or any(r["status"] != "ok" for r in records)):
                    raise ValueError(f"Incomplete/duplicate/failed query execution: {case}")
                logs = sorted((case / "log_dir").glob("*.log"))
                label_condition, metric = condition, "value_bytes_v2"
            if not logs:
                raise ValueError(f"No logs: {case}")
            cases.append((workload, condition, queries, case, logs, metadata, label_condition, metric))
    if pending:
        raise NotReady("; ".join(pending))

    summary, per_query, per_path, per_execution, audit = [], [], [], [], []
    for workload, condition, queries, case, logs, metadata, label_condition, metric in cases:
        data, excluded = parse_logs(logs, queries, label_condition, metric)
        counter = collections.Counter()
        for c in data.values():
            counter.update(c)
        def row_for(c):
            return dict(total_bytes=c["total_bytes"], reader_bytes=c["read_bytes"],
                        cached_bytes=c["total_bytes"]-c["read_bytes"],
                        byte_hit_pct=percentage(c["total_bytes"]-c["read_bytes"], c["total_bytes"]),
                        full_batch_hit_pct=percentage(c["full_batches"], c["payload_batches"]),
                        trace_batches=c["trace_batches"], payload_batches=c["payload_batches"],
                        full_batches=c["full_batches"], metric=metric)
        summary.append(dict(workload=workload, condition=condition, query_count=len(queries),
                            executions="2,3", independent_runs=1, **row_for(counter)))
        for q in sorted(queries):
            c = collections.Counter()
            for (qq, ex, path), value in data.items():
                if qq == q:
                    c.update(value)
            per_query.append(dict(workload=workload, condition=condition, query=f"q{q}", **row_for(c)))
        for path in sorted({p for _, _, p in data}):
            c = collections.Counter()
            for (_, _, pp), value in data.items():
                if pp == path:
                    c.update(value)
            per_path.append(dict(workload=workload, condition=condition, path=path, **row_for(c)))
        for ex in (2, 3):
            c = collections.Counter()
            for (_, ee, _), value in data.items():
                if ee == ex:
                    c.update(value)
            per_execution.append(dict(workload=workload, condition=condition, execution=ex, **row_for(c)))
        audit.append(dict(workload=workload, condition=condition, case=str(case.resolve()),
                          metadata=metadata, measured_queries=sorted(queries), excluded_metrics=dict(excluded),
                          logs=[dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                for p in logs]))
    return summary, per_query, per_path, per_execution, audit


def save(fig, prefix):
    for extension in ("png", "svg", "pdf"):
        path = Path(str(prefix) + "." + extension)
        fig.savefig(path, dpi=220, facecolor="white")
        print(path, flush=True)
    plt.close(fig)


def render(summary, per_query, prefix, source):
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 15,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "hatch.linewidth": 0.6})
    values = {(r["workload"], r["condition"]): r for r in summary}
    note = ("Legacy Parquet-scan traces; ClickBench: 41 queries (q28/q29 not measured)."
            if source == "legacy" else "Decoded value bytes; TPC-H: 22 queries, ClickBench: all 43 queries.")
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.22)
    max_value = 0
    for workload, marker in (("th", "o"), ("cb", "s")):
        ys = [values[workload, c]["byte_hit_pct"] for c in CONDS]
        if any(y is None for y in ys):
            raise ValueError("No payload denominator; cannot draw a zero-valued hit curve")
        ax.plot((0, 1), ys, marker=marker, color=COLORS[workload], linewidth=2.5,
                markersize=10, label=TITLES[workload], zorder=3)
        max_value = max(max_value, *ys)
        for x, y in enumerate(ys):
            dy, va = (-1.0, "top") if workload == "th" else (1.0, "bottom")
            if y < 2 and dy < 0:
                dy, va = 1, "bottom"
            ax.text(x, y + dy, f"{y:.1f}%", color=COLORS[workload], ha="center", va=va,
                    fontsize=14, fontstyle="italic")
    ax.set_xticks((0, 1), ("paging", "paging + reorder\n+ future eviction"))
    ax.set_ylabel("Cache hit ratio (% of traced bytes)")
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(0, max(32, max_value * 1.22))
    ax.grid(axis="y", color="#d6d6d6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.5, 0.055, note + "\nExecutions 2 and 3; one warmup execution excluded.",
             ha="center", fontsize=8.5)
    save(fig, prefix)

    # Explicit per-query denominators permit real 100% hits without hiding large misses.
    data = {(r["workload"], r["query"]): r for r in per_query if r["condition"] == "reorder_future"}
    fig, axes = plt.subplots(2, 1, figsize=(17.0, 8.2))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.12, hspace=0.52)
    for ax, workload, count in zip(axes, ("th", "cb"), (22, 43)):
        x = np.arange(count)
        byte, batch = [], []
        for i in range(1, count+1):
            row = data.get((workload, f"q{i}"))
            byte.append(row["byte_hit_pct"] if row and row["byte_hit_pct"] is not None else np.nan)
            batch.append(row["full_batch_hit_pct"] if row and row["full_batch_hit_pct"] is not None else np.nan)
        ax.bar(x-.18, byte, .36, facecolor=COLORS[workload], edgecolor="black", linewidth=.4)
        ax.bar(x+.18, batch, .36, facecolor="white", edgecolor=COLORS[workload],
               hatch="////", linewidth=.7)
        for i, y in enumerate(byte):
            if np.isnan(y):
                ax.axvspan(i-.4, i+.4, color="#ececec", zorder=0)
                ax.text(i, 7, "N/A", fontsize=7, rotation=90, ha="center")
            elif abs(y-100) < 1e-8:
                ax.text(i-.18, 102, "100", ha="center", va="bottom", fontsize=8, color=COLORS[workload])
        ax.set_xticks(x, [f"q{i}" for i in range(1, count+1)], fontsize=9 if count==22 else 7.5)
        ax.set_xlim(-.7, count-.3)
        ax.set_ylim(0, 112)
        ax.set_yticks((0, 25, 50, 75, 100))
        ax.set_ylabel("Hit ratio (%)", fontsize=13)
        ax.set_title(TITLES[workload], fontsize=15)
        ax.grid(axis="y", color="#dedede", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Per-query hits: page caching + reordering + future-aware eviction", fontsize=16)
    fig.legend(handles=[Patch(facecolor="#666666", edgecolor="black", label="Byte-weighted hit"),
                        Patch(facecolor="white", edgecolor="#666666", hatch="////", label="Full-batch hit")],
               loc="upper center", bbox_to_anchor=(.5, .945), ncol=2, frameon=False, fontsize=12)
    fig.text(.5, .045, note + "\nFull-batch hit = fully cached nonempty batches / all traced nonempty batches.",
             ha="center", fontsize=10)
    save(fig, Path(str(prefix)+"_per_query"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("legacy", "full"), default="legacy")
    parser.add_argument("--input-root", type=Path, default=ROOT / "experiment/FULL_CACHE_PIN_20261007/runs")
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--watch", action="store_true", help="Wait read-only for all four hit cases, then render once")
    args = parser.parse_args()
    if args.watch and args.source != "full":
        parser.error("--watch requires --source full")
    prefix = args.output_prefix or ROOT / f"result/cache_hit_requested_{args.source}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    status_path = Path(str(prefix)+"_status.json")
    deadline, last_pending = time.monotonic()+8*3600, None
    while True:
        try:
            summary, per_query, per_path, per_execution, audit = collect(args.source, args.input_root)
            break
        except NotReady as exc:
            write_json(status_path, dict(status="waiting", source=args.source, pending=str(exc), updated_unix=time.time()))
            if not args.watch:
                print(f"Not ready: {exc}", flush=True)
                return 3
            if str(exc) != last_pending:
                print(f"Waiting: {exc}", flush=True)
                last_pending = str(exc)
            supervisor = args.input_root.parent / "background_status.json"
            if supervisor.exists():
                state = json.loads(supervisor.read_text())
                if state.get("stage") in {"failed", "error", "fatal"}:
                    write_json(status_path, dict(status="error", source=args.source, error=str(state), updated_unix=time.time()))
                    raise ValueError(f"Experiment supervisor failed: {state}")
            if time.monotonic() >= deadline:
                write_json(status_path, dict(status="error", source=args.source,
                    error="Hit cases did not complete within 8 hours", updated_unix=time.time()))
                raise TimeoutError("Hit cases did not complete within 8 hours")
            time.sleep(30)
        except Exception as exc:
            write_json(status_path, dict(status="error", source=args.source, error=str(exc), updated_unix=time.time()))
            raise
    for suffix, rows in (("", summary), ("_per_query", per_query), ("_per_path", per_path),
                         ("_per_execution", per_execution)):
        write_csv(Path(str(prefix)+suffix+".csv"), rows)
    write_json(Path(str(prefix)+"_provenance.json"), dict(source=args.source, cases=audit,
        byte_hit_definition="sum(total_bytes-read_bytes)/sum(total_bytes); pooled executions 2,3",
        full_batch_definition="fully cached nonempty batches / all traced nonempty batches",
        scope=("Legacy normal Parquet materialization only; strings include offsets; late refetch and cached providers may be absent"
               if args.source=="legacy" else "value_bytes_v2: value data and STRING chars, all parquet/cached_provider/late_refetch paths; no offsets/masks/padding")))
    render(summary, per_query, prefix, args.source)
    write_json(status_path, dict(status="complete", source=args.source, updated_unix=time.time(),
                                summary=summary, output_prefix=str(prefix.resolve())))
    for row in summary:
        print(json.dumps(row), flush=True)
    print("100% byte-hit queries:", [(r["workload"],r["query"]) for r in per_query
                                     if r["condition"]=="reorder_future" and r["byte_hit_pct"]==100])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
