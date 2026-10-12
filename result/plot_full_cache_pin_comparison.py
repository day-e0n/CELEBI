#!/usr/bin/env python3
"""Plot completed full-workload cache/pin cases without loading an engine.

  pixi run -e duckdb-python python result/plot_full_cache_pin_comparison.py \
      --input-root experiment/full_cache_pin_20261007

Latency comes from trace-off cases; hit ratios come from separate trace-on cases.
Only complete, comparison_valid cases and executions 2-3 enter reported means.
Missing, failed, unattempted, and absent/zero-payload traces remain N/A, not zero.
Only metric=value_bytes_v2 is included: value data/STRING characters, excluding
offsets, masks, padding, virtual columns and compressed/physical I/O bytes.
Older [cache-hit] lines are recognized and counted as excluded legacy records.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

LABEL = re.compile(r"sirius_set_query_label\('([^']*)'\)")
QUERY = re.compile(r"(.+)_q(\d+)_exec(\d+)$")
FIELD = re.compile(r"(\w+)=('(?:[^']*)'|\S+)")
METRIC = "value_bytes_v2"
EXECS = (2, 3)
PATHS = ("parquet", "cached_provider", "late_refetch")
CONDITIONS = ("baseline", "paging_lru", "reorder_lru", "reorder_future", "pin_gpu")
NAMES = {"baseline": "Baseline", "paging_lru": "Paging",
         "reorder_lru": "Paging + reorder", "reorder_future": "Reorder + future eviction",
         "pin_gpu": "Static GPU pin"}
COLORS = {"baseline": "#909c99", "paging_lru": "#a8c3bf", "reorder_lru": "#3d6b66",
          "reorder_future": "#b4522e", "pin_gpu": "#71648a"}
TITLES = {"th": "TPC-H SF50 (22 queries)", "cb": "ClickBench (43 queries)"}
GIB = 2 ** 30


def mean(values):
    return statistics.mean(values) if values else None


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def ratio(counter):
    return 100 * (1 - counter["read"] / counter["total"]) if counter["total"] else None


def read_traces(case: Path):
    """Associate records with explicit query labels, clearing labels at setup/cleanup."""
    data = collections.defaultdict(collections.Counter)
    excluded = collections.Counter()
    for log in sorted((case / "log_dir").glob("*.log")):
        active = None
        for line in log.open(errors="replace"):
            label = LABEL.search(line)
            if label:
                match = QUERY.fullmatch(label[1])
                active = (match[1], int(match[2]), int(match[3])) if match else None
                continue
            if "[cache-hit]" not in line or active is None or active[2] not in EXECS:
                continue
            fields = {key: value.strip("'") for key, value in FIELD.findall(line)}
            if fields.get("metric") != METRIC:
                excluded["legacy_records"] += 1
                continue
            try:
                total, read = int(fields["total_bytes"]), int(fields["read_bytes"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Malformed cache-hit record in {log}: {line.strip()}") from exc
            if total < 0 or not 0 <= read <= total:
                raise ValueError(f"Invalid payload byte counts in {log}: {line.strip()}")
            path = fields.get("path", "unknown")
            c = data[(active[0], active[1], active[2], path)]
            c["total"] += total
            c["read"] += read
            c["trace_batches"] += 1
            if total:
                c["payload_batches"] += 1
                c["full_payload_batches"] += fields.get("kind") == "full"
    return data, excluded


def write_csv(path: Path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        fields = ({"query_status.csv": ["workload", "condition", "trace_mode", "run", "execution",
                                        "position", "query", "label", "warmup", "status", "wall_s",
                                        "rows", "error", "started_utc", "ended_utc", "case_status",
                                        "included_case", "metadata"],
                   "per_query_path.csv": ["workload", "condition", "query", "path", "hit_pct",
                                          "payload_bytes", "reader_bytes", "cached_bytes",
                                          "trace_batches", "metric"]}.get(path.name, ["status"]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(fig, out: Path, name: str):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=240)
    plt.close(fig)


def bars(ax, values, conditions, ylabel, errors=None, percent=False):
    for x, condition in enumerate(conditions):
        value = values[x]
        if value is None:
            ax.text(x, 0.03, "N/A", transform=ax.get_xaxis_transform(), ha="center", fontsize=7)
            continue
        err = errors[x] if errors else None
        ax.bar(x, value, color=COLORS[condition], edgecolor="#2f3b3a", linewidth=.6,
               yerr=err, capsize=2, zorder=2)
        label = f"{value:.1f}%" if percent else f"{value:.2f}"
        ax.annotate(label, (x, value + (err or 0)), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7)
    ax.set_xticks(range(len(conditions)), [NAMES[c].replace(" + ", "\n+ ") for c in conditions], fontsize=6.5)
    ax.set_xlim(-.65, len(conditions) - .35)
    if percent:
        ax.set_ylim(0, 110)
    elif any(v is not None for v in values):
        ax.set_ylim(0, max(v + (errors[i] or 0 if errors else 0)
                           for i, v in enumerate(values) if v is not None) * 1.2 or 1)
    else:
        ax.set_ylim(0, 1)
    ps.finish(ax, ylabel)


def collect(manifest: dict):
    queries = {w: [int(q) for q in plan["arrival_order"]] for w, plan in manifest["workloads"].items()}
    walls = collections.defaultdict(list)
    query_walls = collections.defaultdict(list)
    hits = collections.defaultdict(collections.Counter)
    query_hits = collections.defaultdict(collections.Counter)
    path_hits = collections.defaultdict(collections.Counter)
    query_paths = collections.defaultdict(collections.Counter)
    hit_execs = collections.defaultdict(set)
    hit_queries = collections.defaultdict(set)
    pin_times = collections.defaultdict(list)
    statuses, query_records = [], []
    for name in manifest["cases"]:
        metadata = Path(name)
        meta = json.loads(metadata.read_text())
        case = Path(meta["case_directory"])
        w, cond, mode = meta["workload"], meta["condition"], meta["trace_mode"]
        run = int(meta["run"])
        key = w, cond
        csv_path = Path(meta.get("query_csv", str(case / "queries.csv")))
        rows = list(csv.DictReader(csv_path.open())) if csv_path.exists() else []
        wanted = {(ex, q) for ex in EXECS for q in queries[w]}
        observed = [(int(r["execution"]), int(r["query"])) for r in rows
                    if int(r["execution"]) in EXECS]
        usable = (meta.get("comparison_valid") is True and meta.get("status") == "complete"
                  and len(observed) == len(wanted) and set(observed) == wanted
                  and all(r["status"] == "ok" and r["wall_s"] not in ("", None)
                          for r in rows if int(r["execution"]) in EXECS))
        trace_data, excluded, trace_error = {}, {}, ""
        if mode == "hit":
            try:
                trace_data, excluded = read_traces(case)
            except ValueError as exc:
                trace_error, usable = str(exc), False
        counts = collections.Counter(r["status"] for r in rows)
        statuses.append({"workload": w, "condition": cond, "trace_mode": mode, "run": run,
                         "status": meta.get("status", "unknown"), "included": usable,
                         "ok": counts["ok"], "error": counts["error"], "fatal": counts["fatal"],
                         "unattempted": counts["unattempted"], "returncode": meta.get("returncode"),
                         "timed_out": meta.get("timed_out"), "pin_setup_complete": meta.get("pin_setup_complete"),
                         "pin_preparation_wall_s": meta.get("pin_preparation_wall_s"),
                         "legacy_records_excluded": excluded.get("legacy_records", 0),
                         "trace_error": trace_error, "metadata": str(metadata)})
        for row in rows:
            query_records.append({**row, "case_status": meta.get("status", "unknown"),
                                  "included_case": usable, "metadata": str(metadata)})
        if not usable:
            continue
        if mode == "latency":
            for ex in EXECS:
                execution_rows = [r for r in rows if int(r["execution"]) == ex]
                walls[key].append(sum(float(r["wall_s"]) for r in execution_rows))
                for row in execution_rows:
                    query_walls[(w, cond, int(row["query"]))].append(float(row["wall_s"]))
            if cond == "pin_gpu" and meta.get("pin_setup_complete") is True:
                value = meta.get("pin_preparation_wall_s")
                if value is not None:
                    pin_times[w].append(float(value))
        if mode != "hit":
            continue
        for (label_condition, q, ex, path), counter in trace_data.items():
            if label_condition != cond or q not in queries[w]:
                raise ValueError(f"Trace label does not match case: {metadata}, {label_condition}, q{q}")
            hits[key].update(counter)
            query_hits[(w, cond, q)].update(counter)
            path_hits[(w, cond, path)].update(counter)
            query_paths[(w, cond, q, path)].update(counter)
            hit_queries[key].add((run, ex, q))
        for ex in EXECS:
            hit_execs[key].add((run, ex))
    return queries, walls, query_walls, hits, query_hits, path_hits, query_paths, hit_execs, hit_queries, pin_times, statuses, query_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--input-root", dest="input_root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_path = args.input_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    out = args.output or args.input_root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    (queries, walls, query_walls, hits, query_hits, path_hits, query_paths, hit_execs,
     hit_queries, pin_times, statuses, query_records) = collect(manifest)
    workloads = [w for w in ("th", "cb") if w in queries]
    conditions = [c for c in CONDITIONS if c in manifest["conditions"]]
    overall, per_query, per_path, per_query_path = [], [], [], []
    for w in workloads:
        for c in conditions:
            key, payload = (w, c), hits[(w, c)]
            n = len(hit_execs[key])
            observed_payload = bool(payload["trace_batches"])
            overall.append({"workload": w, "condition": c, "query_count": len(queries[w]),
                            "latency_samples": len(walls[key]), "wall_mean_s": mean(walls[key]),
                            "wall_sd_s": sd(walls[key]), "hit_executions": n,
                            "hit_pct": ratio(payload), "payload_gib_per_execution": payload["total"] / GIB / n if n and observed_payload else None,
                            "reader_gib_per_execution": payload["read"] / GIB / n if n and observed_payload else None,
                            "cached_gib_per_execution": (payload["total"] - payload["read"]) / GIB / n if n and observed_payload else None,
                            "traced_query_samples": len(hit_queries[key]), "expected_query_samples": n * len(queries[w]),
                            "trace_batches": payload["trace_batches"], "metric": METRIC})
            for q in sorted(queries[w]):
                values, payload = query_walls[(w, c, q)], query_hits[(w, c, q)]
                observed_payload = bool(payload["trace_batches"])
                per_query.append({"workload": w, "condition": c, "query": f"q{q}",
                                  "latency_samples": len(values), "wall_mean_s": mean(values), "wall_sd_s": sd(values),
                                  "hit_pct": ratio(payload), "payload_bytes": payload["total"] if observed_payload else None,
                                  "reader_bytes": payload["read"] if observed_payload else None,
                                  "cached_bytes": payload["total"] - payload["read"] if observed_payload else None,
                                  "trace_batches": payload["trace_batches"], "metric": METRIC})
            for path in sorted(set(PATHS) | {p for ww, cc, p in path_hits if (ww, cc) == key}):
                payload = path_hits[(w, c, path)]
                observed_payload = bool(payload["trace_batches"])
                per_path.append({"workload": w, "condition": c, "path": path, "hit_pct": ratio(payload),
                                 "payload_bytes": payload["total"] if observed_payload else None,
                                 "reader_bytes": payload["read"] if observed_payload else None,
                                 "cached_bytes": payload["total"] - payload["read"] if observed_payload else None,
                                 "trace_batches": payload["trace_batches"],
                                 "metric": METRIC})
                for q in sorted(queries[w]):
                    payload = query_paths[(w, c, q, path)]
                    if not payload["trace_batches"]:
                        continue
                    per_query_path.append({"workload": w, "condition": c, "query": f"q{q}", "path": path,
                                           "hit_pct": ratio(payload), "payload_bytes": payload["total"],
                                           "reader_bytes": payload["read"], "cached_bytes": payload["total"] - payload["read"],
                                           "trace_batches": payload["trace_batches"], "metric": METRIC})
    for filename, rows in (("overall.csv", overall), ("per_query.csv", per_query), ("per_path.csv", per_path),
                           ("per_query_path.csv", per_query_path), ("case_status.csv", statuses),
                           ("query_status.csv", query_records)):
        write_csv(out / filename, rows)
    pin_rows = [{"workload": w, "samples": len(pin_times[w]), "pin_preparation_mean_s": mean(pin_times[w]),
                 "pin_preparation_sd_s": sd(pin_times[w])} for w in workloads]
    write_csv(out / "pin_setup.csv", pin_rows)
    by_case = {(r["workload"], r["condition"]): r for r in overall}

    fig, axes = plt.subplots(len(workloads), 2, figsize=(10.5, 3.3 * len(workloads)), squeeze=False)
    for row, w in enumerate(workloads):
        data = [by_case[(w, c)] for c in conditions]
        bars(axes[row, 0], [r["wall_mean_s"] for r in data], conditions, "Whole-workload wall time (s)",
             [r["wall_sd_s"] for r in data])
        bars(axes[row, 1], [r["hit_pct"] for r in data], conditions, "Cached value payload (%)", percent=True)
        axes[row, 0].set_title(f"{TITLES[w]} — trace off")
        axes[row, 1].set_title(f"{TITLES[w]} — trace on")
    fig.suptitle("Full workload including q24 · executions 2–3 · complete cases only\n"
                 "Latency and hit measurements are separate runs; hit denominator is traced scan requests.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, out, "overall")

    fig, axes = plt.subplots(len(workloads), 1, figsize=(13.5, 2.7 * len(workloads)), squeeze=False)
    for row, w in enumerate(workloads):
        ax, qs = axes[row, 0], sorted(queries[w])
        matrix = [[ratio(query_hits[(w, c, q)]) for q in qs] for c in conditions]
        grid = [[v if v is not None else math.nan for v in values] for values in matrix]
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#e7e7e7")
        im = ax.imshow(grid, aspect="auto", vmin=0, vmax=100, cmap=cmap)
        ax.set_xticks(range(len(qs)), [f"q{q}" for q in qs], rotation=90, fontsize=6)
        ax.set_yticks(range(len(conditions)), [NAMES[c] for c in conditions], fontsize=7)
        for y, values in enumerate(matrix):
            for x, value in enumerate(values):
                ax.text(x, y, "N/A" if value is None else f"{value:.0f}", ha="center", va="center",
                        fontsize=5, color="white" if value is not None and value > 65 else "#222222")
        for tick, q in zip(ax.get_xticklabels(), qs):
            if q == 24:
                tick.set_color(COLORS["reorder_future"])
                tick.set_fontweight("bold")
        ax.set_title(f"{TITLES[w]} — byte-weighted hit by query (trace on)")
        fig.colorbar(im, ax=ax, label="Cached value payload (%)", fraction=.018, pad=.01)
    fig.tight_layout()
    save(fig, out, "per_query_hit")

    fig, axes = plt.subplots(len(workloads), 1, figsize=(13.5, 3 * len(workloads)), squeeze=False)
    for row, w in enumerate(workloads):
        ax, qs = axes[row, 0], sorted(queries[w])
        for c in conditions:
            values = [mean(query_walls[(w, c, q)]) for q in qs]
            ax.plot(range(len(qs)), [v if v is not None else math.nan for v in values],
                    color=COLORS[c], marker=".", linewidth=1, label=NAMES[c])
        ax.set_xticks(range(len(qs)), [f"q{q}" for q in qs], rotation=90, fontsize=6)
        ax.set_title(f"{TITLES[w]} — per-query mean wall (trace off; missing cases are gaps)")
        ps.finish(ax, "Wall time (s)")
    axes[0, 0].legend(handles=[Line2D([], [], color=COLORS[c], label=NAMES[c]) for c in conditions],
                      ncol=len(conditions), fontsize=7, loc="upper center")
    fig.tight_layout()
    save(fig, out, "per_query_latency")

    fig, axes = plt.subplots(len(workloads), len(PATHS), figsize=(13.5, 3.1 * len(workloads)), squeeze=False)
    for row, w in enumerate(workloads):
        for col, path in enumerate(PATHS):
            ax = axes[row, col]
            bars(ax, [ratio(path_hits[(w, c, path)]) for c in conditions], conditions,
                 "Cached value payload (%)", percent=True)
            ax.set_title(f"{TITLES[w]}\n{path}")
    fig.suptitle("Payload coverage by serving path · N/A = no traced payload for that path", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, out, "paths")

    fig, ax = plt.subplots(figsize=(5, 3))
    for x, w in enumerate(workloads):
        value = mean(pin_times[w])
        if value is None:
            ax.text(x, .03, "N/A", transform=ax.get_xaxis_transform(), ha="center")
        else:
            ax.bar(x, value, yerr=sd(pin_times[w]), color=COLORS["pin_gpu"], capsize=2)
            ax.annotate(f"{value:.2f}", (x, value), xytext=(0, 4), textcoords="offset points", ha="center")
    ax.set_xticks(range(len(workloads)), [TITLES[w] for w in workloads], fontsize=7)
    ax.set_xlim(-.6, len(workloads) - .4)
    if not any(pin_times[w] for w in workloads):
        ax.set_ylim(0, 1)
    ps.finish(ax, "Static pin preparation (s)")
    ax.set_title("Separate initialization cost · completed trace-off pin cases")
    fig.tight_layout()
    save(fig, out, "pin_setup")

    def display(value, digits=2):
        return "N/A" if value is None else f"{value:.{digits}f}"

    text = ["# Full-workload cache and GPU pin comparison", "",
            f"Input: `{manifest_path.resolve()}`. Executions 2–3; execution 1 is warmup.", "",
            "Only complete cases marked comparison_valid enter the reported numbers. N/A is never replaced by zero.",
            "Latency is from trace-off cases; hit coverage is from separate trace-on cases. Error bars show sample SD.",
            "Hit = 1 − sum(reader value bytes)/sum(request value bytes), weighted across all traced requests.",
            "value_bytes_v2 counts fixed-width values and STRING characters; offsets, masks, virtual columns, padding, slabs, and physical I/O are excluded.",
            "A query with no traced payload has N/A coverage. The overall coverage describes traced requests; traced-query counts are in overall.csv.",
            "All 22 TPC-H / 43 ClickBench queries are retained, including ClickBench q24, q28, q29.", "",
            "| Workload | Condition | Wall mean ± SD (s) | Latency samples | Cached payload | Hit executions |",
            "|---|---|---:|---:|---:|---:|"]
    for r in overall:
        wall = "N/A" if r["wall_mean_s"] is None else f"{display(r['wall_mean_s'])} ± {display(r['wall_sd_s'])}"
        hit = "N/A" if r["hit_pct"] is None else f"{display(r['hit_pct'])}%"
        text.append(f"| {TITLES[r['workload']]} | {NAMES[r['condition']]} | {wall} | {r['latency_samples']} | {hit} | {r['hit_executions']} |")
    text += ["", "Pin initialization is reported separately in pin_setup.csv and pin_setup.{png,svg,pdf}.", "",
             "| Workload | Mode | Run | Condition | Status | Included | Error / fatal / unattempted |",
             "|---|---|---:|---|---|---|---|"]
    for r in statuses:
        text.append(f"| {r['workload']} | {r['trace_mode']} | {r['run']} | {r['condition']} | {r['status']} | {r['included']} | {r['error']} / {r['fatal']} / {r['unattempted']} |")
    legacy = sum(r["legacy_records_excluded"] for r in statuses)
    text += ["", f"Legacy cache-hit records excluded: {legacy}. Trace parse errors, return codes and timeout flags are preserved in case_status.csv.",
             "Raw query status/error text and partial successes are retained in query_status.csv; they are not cherry-picked into headline means.",
             "Static pin is a fixed column subset within the memory budget, not whole-dataset pin; see the frozen manifest for the selection.", ""]
    (out / "summary.md").write_text("\n".join(text))
    print(f"Wrote figures (PNG/SVG/PDF), CSVs and summary: {out.resolve()}")
    print(f"Included {sum(r['included'] for r in statuses)}/{len(statuses)} cases; legacy records excluded={legacy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
