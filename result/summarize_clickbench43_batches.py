#!/usr/bin/env python3
"""Audit and export the actual 43-query, 1/2/4-batch comparison."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "result"))
sys.path.insert(0, str(ROOT / "scripts"))
from plot_requested_cache_hits import parse_logs  # noqa: E402
from run_clickbench43_batch_comparison import local_position  # noqa: E402
from run_full_cache_pin_comparison import digest  # noqa: E402
import cache_aware_query_reorder as reorder  # noqa: E402
from future_cache_hints import hint_spec  # noqa: E402

CONDITIONS = ("baseline", "paging_lru", "reorder_future")
EXPECTED = {(ex, q) for ex in (1, 2, 3) for q in range(1, 44)}


def audited_case(path):
    meta = json.loads(path.read_text())
    if meta["status"] != "complete" or meta.get("comparison_valid") is not True:
        raise ValueError(f"Measurement is not complete: {path}, {meta['status']}")
    records = list(csv.DictReader(Path(meta["query_csv"]).open()))
    actual = {(int(r["execution"]), int(r["query"])) for r in records}
    if (
        len(records) != len(EXPECTED)
        or actual != EXPECTED
        or any(r["status"] != "ok" for r in records)
    ):
        raise ValueError(f"Incomplete or duplicate 43-query measurement: {path}")
    events = [
        json.loads(line) for line in Path(meta["journal"]).read_text().splitlines()
    ]
    assert any(e["stage"] == "worker_finished" for e in events)
    if "batch_orders" in meta and meta["condition"] == "reorder_future":
        hints = [e for e in events if e["stage"] == "batch_hint"]
        assert len(hints) == len(EXPECTED)
        reorder.load_column_set("clickbench_pin_columns")
        for event in hints:
            batch = meta["batch_orders"][event["batch"] - 1]
            local = event["local_position"]
            assert event["query"] == batch[local - 1]
            assert event["hint"] == hint_spec(
                "evict", batch, 3, event["execution"], local, "execution"
            )
        for ex in (1, 2, 3):
            executed = [int(r["query"]) for r in records if int(r["execution"]) == ex]
            assert executed == meta["order"]
            for pos, q in enumerate(executed, 1):
                _, batch, local = local_position(meta["batch_orders"], pos)
                assert batch[local - 1] == q
    return meta, records


def latency(path):
    meta, records = audited_case(path)
    assert meta["trace_mode"] == "latency"
    assert meta["environment"]["SIRIUS_CACHE_HIT_TRACE"] == "0"
    totals = [
        sum(float(r["wall_s"]) for r in records if int(r["execution"]) == ex)
        for ex in (2, 3)
    ]
    value = statistics.mean(totals)
    assert math.isclose(value, meta["steady_state_workload_wall_s"], abs_tol=1e-9)
    return value, dict(
        metadata=str(path),
        execution_2_s=totals[0],
        execution_3_s=totals[1],
        mean_s=value,
    )


def hits(path):
    meta, _ = audited_case(path)
    assert meta["trace_mode"] == "hit"
    assert meta["environment"]["SIRIUS_CACHE_HIT_TRACE"] == "1"
    logs = sorted((Path(meta["case_directory"]) / "log_dir").glob("*.log"))
    data, excluded = parse_logs(
        logs, set(range(1, 44)), meta["condition"], "value_bytes_v2"
    )
    count = Counter()
    for part in data.values():
        count.update(part)
    total = count["total_bytes"]
    cached = total - count["read_bytes"]
    ratio = 100 * cached / total
    return ratio, dict(
        metadata=str(path),
        logs=list(map(str, logs)),
        excluded_metrics=dict(excluded),
        total_bytes=total,
        reader_bytes=count["read_bytes"],
        cached_bytes=cached,
        byte_hit_pct=ratio,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "experiment/BATCH43_20261007"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "result/clickbench43_batch_1_2_4"
    )
    args = parser.parse_args()
    queue = json.loads((args.input / "job_status.json").read_text())
    if queue["status"] != "complete" or queue["overlaps"]:
        raise ValueError(
            "Batch measurements pending, failed, or overlapped another GPU job"
        )
    first = ROOT / "experiment/AVG_FIX_CB_20261007"
    rows, details = [], []
    extension_sha = None
    for batches in (1, 2, 4):
        lat_root = (
            first / "latency_runs"
            if batches == 1
            else args.input / f"b{batches}_latency"
        )
        manifest = json.loads((lat_root / "manifest.json").read_text())
        if extension_sha is None:
            extension_sha = manifest["extension_sha256"]
        assert manifest["extension_sha256"] == extension_sha
        assert digest(Path(manifest["extension"])) == extension_sha
        assert manifest["runs"] == 1
        cases = {
            json.loads(Path(p).read_text())["condition"]: Path(p)
            for p in manifest["cases"]
        }
        times, time_audit = {}, {}
        for condition in CONDITIONS:
            times[condition], time_audit[condition] = latency(cases[condition])
        hit_audit, ratios = {}, {}
        for condition in CONDITIONS[1:]:
            if batches == 1:
                folder = (
                    "hit_paging_retry_job"
                    if condition == "paging_lru"
                    else "hit_future_process_exit_job"
                )
                hit_root = first / folder / "runs"
            else:
                hit_root = args.input / f"b{batches}_hit"
            hit_manifest = json.loads((hit_root / "manifest.json").read_text())
            assert hit_manifest["extension_sha256"] == extension_sha
            path = next(
                Path(p)
                for p in hit_manifest["cases"]
                if json.loads(Path(p).read_text())["condition"] == condition
            )
            latency_meta = json.loads(cases[condition].read_text())
            hit_meta = json.loads(path.read_text())
            for key in (
                "order",
                "gpu_device",
                "gpu_usage_limit_bytes",
                "host_capacity_bytes",
                "cache_budget_bytes",
                "min_free_bytes_per_gpu",
                "future_mode",
                "future_horizon",
                "late_materialization",
                "prefetch",
                "odirect",
            ):
                assert latency_meta[key] == hit_meta[key], (batches, condition, key)
            if batches > 1:
                assert latency_meta["batch_orders"] == hit_meta["batch_orders"]
            ratios[condition], hit_audit[condition] = hits(path)
        sizes = [43] if batches == 1 else manifest["batch_sizes"]
        row = dict(
            workload="ClickBench",
            queries=43,
            batches=batches,
            queries_per_batch="+".join(map(str, sizes)),
            baseline_s=times["baseline"],
            page_s=times["paging_lru"],
            celebi_s=times["reorder_future"],
            page_hit_pct=ratios["paging_lru"],
            celebi_hit_pct=ratios["reorder_future"],
            reduction_pct=100 * (1 - times["reorder_future"] / times["baseline"]),
        )
        rows.append(row)
        details.append(
            dict(
                batches=batches,
                queries_per_batch=sizes,
                latency=time_audit,
                hit=hit_audit,
            )
        )
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tex = [
        r"% Requires \usepackage{booktabs}.",
        r"{\footnotesize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Workload & \multicolumn{3}{c}{Latency (s)} & \shortstack{Reduction\\(\%)} \\",
        r"\cmidrule(lr){2-4}",
        r"& Base & Page & CELEBI & CELEBI \\",
        r"\midrule",
    ]
    for row in rows:
        tex.append(
            f"ClickBench ($B={row['batches']}$) & {row['baseline_s']:.2f} & {row['page_s']:.2f} & \\textbf{{{row['celebi_s']:.2f}}} & \\textbf{{{row['reduction_pct']:.2f}}} \\\\"
        )
    tex.extend(
        [
            r"\midrule",
            r"TPC-H SF50 ($B=1$) & 32.56 & 30.75 & \textbf{23.82} & \textbf{26.86} \\",
            r"\bottomrule",
            r"\end{tabular}",
            "}",
            "% ClickBench: all 43 queries, including q28/q29. B is the number of query batches.",
            "% B=1: 43 queries; B=2: 22+21; B=4: 11+11+11+10. Batch split seed=1.",
            "% Each row sums ALL 43 per-query execute().fetchall() times, averaged over executions 2 and 3.",
            "% TPC-H B=1 reuses the previous measurement; no new TPC-H batch measurement was performed.",
        ]
    )
    out.with_suffix(".tex").write_text("\n".join(tex) + "\n")
    provenance = dict(
        extension_sha256=extension_sha,
        independent_runs_per_condition=1,
        batch_split_seed=1,
        executions=[2, 3],
        warmup_execution=1,
        all_43_queries_in_every_case=True,
        cache_reset_between_batches=False,
        future_horizon="remaining queries in the current batch only",
        latency_metric="sum of all 43 execute().fetchall() elapsed times, excluding label/hint calls and reorder preparation",
        hit_metric="value_bytes_v2: cache-served/requested decoded value bytes pooled over executions 2 and 3; repeated accesses counted; excludes offsets, masks, padding",
        post_workload_exit_policy="After ALL queries succeed, CUDA synchronization and C/Python log flushing precede isolated process exit, bypassing only the frozen engine's final destructor",
        queue=queue,
        results=rows,
        sources=details,
        reused_tpch_source=str(
            ROOT / "result/query_latency_and_cache_hit_after_avg_fix_provenance.json"
        ),
    )
    out.with_name(out.name + "_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    for row in rows:
        print(json.dumps(row))
    print(out.with_suffix(".tex"))


if __name__ == "__main__":
    main()
