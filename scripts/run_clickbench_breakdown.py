#!/usr/bin/env python3
"""ClickBench 43-query latency breakdown, same 4 conditions as
run_random22_breakdown_var.py:

  baseline              -- no caching at all
  celebi_fixed          -- fixed-width page cache + reorder
  celebi_variable       -- variable-width (STRING) page cache + reorder
  celebi_fixed_variable -- both page caches + reorder

Why ClickBench rather than TPC-H/JCC-H for the page-cache work: it is a single
denormalized table, so no scan ever carries a join's dynamic filter -- the gate
that excluded 675 of 795 lineitem scans from the cache on SF50 simply has
nothing to fire on. And its queries are narrow (median 2 columns of 105), so a
typical cache entry is 0.2-1.6 GB at 100M rows instead of the 3.4-14.8 GiB
lineitem projections that every one of TPC-H's 18 distinct keys exceeded the
per-entry admission cap with.

The three caching conditions reuse the SAME reordered sequence (computed once
from the fixed-width-scope reorder) so only the cache configuration differs.
"""

from __future__ import annotations

import argparse
import os
import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TPCH_DIR = REPO_ROOT / "test" / "tpch_performance"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(TPCH_DIR))
from performance_test import load_query_set  # noqa: E402
from queries import QUERIES  # noqa: E402
import cache_aware_query_reorder as reorder  # noqa: E402
from run_random22_breakdown import write_config  # noqa: E402
from future_cache_hints import hint_spec  # noqa: E402
from run_random22_breakdown_var import CONDITIONS, make_env  # noqa: E402

import duckdb  # noqa: E402

# Arrival order = the benchmark's own query order, so a reorder's benefit is
# measured against the sequence a user would actually submit.
ARRIVAL_ORDER = list(range(1, 44))


def _budget_bytes(text: str) -> int:
    """"6GB" -> bytes, for the byte-lru reorder's simulated cache size.

    Mirrors what SIRIUS_FIXED_PAGE_CACHE_BYTES_PER_GPU is set to, so the
    simulation evicts at the same point the real cache does.
    """

    value = text.strip().upper()
    for suffix, scale in (("GB", 2 ** 30), ("MB", 2 ** 20), ("KB", 2 ** 10), ("B", 1)):
        if value.endswith(suffix):
            return int(float(value[: -len(suffix)]) * scale)
    return int(value)


def open_clickbench_connection(parquet_path: str, extension_path: str):
    """In-memory DuckDB with `hits` as a view over the single parquet file.

    performance_test.open_connection() registers the 8 TPC-H tables from a
    directory; ClickBench is one table from one file, so it gets its own opener
    rather than another special case threaded through that one.
    """
    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    con.execute(
        f"CREATE OR REPLACE VIEW hits AS SELECT * FROM read_parquet('{parquet_path}')"
    )
    threads = os.environ.get("BENCH_DUCKDB_THREADS")
    if threads:
        con.execute(f"SET threads={int(threads)}")
    con.execute(f"LOAD '{extension_path}'")
    return con


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/mnt/nvme/clickbench/hits_v2.parquet")
    parser.add_argument("--executions", type=int, default=3)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--fixed-cache-budget", default="6GB")
    parser.add_argument("--variable-cache-budget", default="4GB")
    parser.add_argument("--variable-page-bytes", default="16777216")
    # Defaults to 0, unlike the TPC-H runner's 4096MB: the memory-pressure sweep
    # that setting arms was measured to free 0 bytes in 29 of 29 events on SF50
    # while being the entire run-to-run variance (+-2756ms -> +-649ms with it off).
    parser.add_argument("--min-free-bytes-per-gpu", default="0")
    parser.add_argument("--gpu-usage-limit", default="20GB")
    parser.add_argument("--arrival", choices=("natural", "worst"), default="natural",
                        help="Arrival order before any reorder. 'worst' minimises the column "
                             "overlap between adjacent queries, which is the order a reorder has "
                             "the most to recover from; the benchmark's natural order can already "
                             "be favourable and then understates what reordering is worth.")
    parser.add_argument("--order",
                        help="Explicit comma-separated query order. Overrides --arrival but "
                             "NOT the reorder, which still runs on top of it; for pinning an "
                             "arrival order the built-in policies cannot express.")
    parser.add_argument("--reorder-policy", choices=("fixed-overlap", "byte-overlap", "byte-lru", "cost-ascending",
                                 "unfiltered-overlap", "fixed-then-variable", "fixed-first", "fixed-bytes-then-variable", "cacheable-first", "resident-overlap", "resident-overlap-spread", "big-same-adjacent", "big-diff-apart", "small-first-big-last", "small-first-big-runs", "small-first-big-ascending", "overlap-fit"),
                        default="fixed-overlap",
                        help="'fixed-overlap' maximises adjacent-pair overlap counted in "
                             "COLUMNS, so a 25-row nation column scores like a 600M-row "
                             "lineitem one and position 1 -- which always misses -- is free. "
                             "'byte-lru' instead maximises the bytes served by a simulated "
                             "byte-budget LRU over the whole sequence; needs --column-bytes.")
    parser.add_argument("--query-table-filters",
                        help="JSON from scripts/extract_query_filters.py, required by "
                             "unfiltered-overlap: which table scans carry a predicate.")
    parser.add_argument("--query-table-columns",
                        help="JSON from scripts/probe_scan_requests.py, required by "
                             "cost-ascending.")
    parser.add_argument("--column-bytes",
                        help="JSON from scripts/probe_column_bytes.py, required by byte-lru.")
    parser.add_argument("--future-cache", choices=("off", "evict", "admit", "bypass"), default="off",
                        help="Tell the page cache the upcoming query order before each query "
                             "(scripts/future_cache_hints.py): 'evict' evicts the column needed "
                             "furthest ahead first, 'admit' also refuses columns not read again, "
                             "'bypass' evicts like 'evict' and skips a column needed later than "
                             "everything resident when the store is full.")
    parser.add_argument("--future-horizon", choices=("execution", "run"), default="execution",
                        help="How far ahead the cache sees: the rest of this execution, or "
                             "every remaining execution of the run.")
    parser.add_argument("--batches", type=int, default=1,
                        help="Split the queries into this many batches at random (--batch-seed); "
                             "each batch keeps the arrival order, is reordered on its own, and "
                             "the page cache is told only the rest of the current batch. The "
                             "batches run back to back in one process, so the cache carries over.")
    parser.add_argument("--batch-seed", type=int, default=0)
    parser.add_argument("--no-reorder", action="store_true",
                        help="Run the arrival order even for a caching condition. The reorder is "
                             "normally applied to every non-baseline condition, which makes a "
                             "cache-vs-baseline comparison measure the CELEBI package rather than "
                             "the cache; this isolates the two.")
    parser.add_argument("--skip-queries", default="",
                        help="Comma-separated query numbers to drop, e.g. '9,28'. One "
                             "query that aborts the engine would otherwise take the "
                             "whole multi-execution run with it.")
    args = parser.parse_args()

    print(f"query set: {load_query_set('clickbench_queries')}", flush=True)
    print(f"column set: {reorder.load_column_set('clickbench_pin_columns')}", flush=True)

    skip = {int(x) for x in args.skip_queries.split(",") if x.strip()}
    qnums = [q for q in ARRIVAL_ORDER if q not in skip]
    if skip:
        print(f"skipping q{', q'.join(str(q) for q in sorted(skip))} "
              f"({len(qnums)} queries per execution)", flush=True)
    if args.order is not None and not args.order.strip():
        # An empty --order used to fall through to --arrival's default and run the
        # natural order, which is favourable to the cache: a whole 24-run sweep was
        # measured against the wrong workload before anyone noticed.
        parser.error("--order was given but is empty")
    if args.order:
        qnums = [int(x) for x in args.order.split(",") if x.strip()]
        print(f"explicit order: {','.join(f'q{q}' for q in qnums)}", flush=True)
    elif args.arrival == "worst":
        qnums = reorder.worst_case_sequence(qnums, "fixed_width")
        print(f"arrival(worst): {','.join(f'q{q}' for q in qnums)}", flush=True)
    if args.batches > 1:
        import random
        shuffled = list(qnums)
        random.Random(args.batch_seed).shuffle(shuffled)
        groups = [shuffled[i::args.batches] for i in range(args.batches)]
        position_of = {q: i for i, q in enumerate(qnums)}
        batches = [sorted(g, key=position_of.__getitem__) for g in groups]
    else:
        batches = [qnums]
    for i, b in enumerate(batches, start=1):
        if len(batches) > 1:
            print(f"batch {i}: {','.join(f'q{q}' for q in b)}", flush=True)
    if args.condition != "baseline" and not args.no_reorder:
        if args.reorder_policy in ("byte-lru", "byte-overlap", "cost-ascending",
                                       "fixed-then-variable", "fixed-bytes-then-variable",
                                       "fixed-first", "cacheable-first", "resident-overlap", "resident-overlap-spread", "big-same-adjacent", "big-diff-apart", "small-first-big-last", "small-first-big-runs", "small-first-big-ascending", "overlap-fit"):
            if not args.column_bytes:
                parser.error(f"--reorder-policy {args.reorder_policy} requires --column-bytes")
            print(f"column bytes: {reorder.load_column_bytes(args.column_bytes)} columns",
                  flush=True)
        if args.reorder_policy == "unfiltered-overlap":
            if not args.query_table_filters:
                parser.error("--reorder-policy unfiltered-overlap requires --query-table-filters")
            print(f"query filters: "
                  f"{reorder.load_query_table_filters(args.query_table_filters)} queries",
                  flush=True)
        if args.reorder_policy == "cost-ascending":
            if not args.query_table_columns:
                parser.error("--reorder-policy cost-ascending requires --query-table-columns")
            print(f"query columns: "
                  f"{reorder.load_query_table_columns(args.query_table_columns)} queries",
                  flush=True)
        cfg = reorder.ReorderConfig(policy=args.reorder_policy, scope="fixed_width",
                                    window=0, keep_first=False,
                                    resident_column_budget=0,
                                    cache_budget_bytes=_budget_bytes(args.fixed_cache_budget))
        batches = [list(reorder.reorder_query_sequence(b, cfg).reordered_queries) for b in batches]
        for i, b in enumerate(batches, start=1):
            print(f"reordered{f' batch {i}' if len(batches) > 1 else ''}: "
                  f"{','.join(f'q{q}' for q in b)}", flush=True)
    qnums = [q for b in batches for q in b]

    output = args.output.resolve()
    case_dir = output / args.condition
    log_dir = case_dir / "log_dir"
    config_path = case_dir / "sirius.yaml"
    write_config(config_path, case_dir / "telemetry_data", args.gpu_usage_limit)
    env = make_env(args.condition, args.devices, log_dir, config_path,
                   args.fixed_cache_budget, args.variable_cache_budget,
                   args.variable_page_bytes, args.min_free_bytes_per_gpu)
    os.environ.update(env)

    # build_operator_breakdown_comparison.py's steady_state_series() only knows
    # the literal labels "cold_hot" and "paging" for "execution 1 is warmup" --
    # reuse them rather than teaching that shared script new condition names.
    series_label = "cold_hot" if args.condition == "baseline" else "paging"

    from performance_test import EXTENSION_PATH
    con = open_clickbench_connection(args.input, EXTENSION_PATH)
    # The coalescer packs row groups into a split until this many UNCOMPRESSED
    # bytes; a small value gives one row group per split, which is what makes a
    # cached chunk identifiable by its row group across queries that project
    # different columns (the byte cap otherwise lands the split boundary in a
    # different place for each projection, and the provenance match fails).
    batch = os.environ.get("BENCH_SCAN_TASK_BATCH_SIZE")
    if batch:
        con.execute(f"SET scan_task_batch_size = {int(batch)}")
        print(f"scan_task_batch_size: {int(batch)}", flush=True)
    try:
        for execution in range(1, args.executions + 1):
            for position, (batch, in_batch, qnum) in enumerate(
                    ((b, j, q) for b in batches for j, q in enumerate(b, start=1)), start=1):
                label = f"{series_label}_q{qnum}_exec{execution}"
                con.execute(f"CALL sirius_set_query_label('{label}')")
                if args.future_cache != "off" and args.condition != "baseline":
                    # With batches, the cache sees only the rest of the current batch.
                    spec = hint_spec(args.future_cache, batch, args.executions, execution,
                                     in_batch,
                                     args.future_horizon if len(batches) == 1 else "execution")
                    con.execute(f"CALL sirius_set_cache_hints('{spec}')")
                con.execute("SET gpu_execution = true;")
                # Wall clock per query. The telemetry's total_ms is summed GPU operator
                # time, which a query can lower while taking longer end to end, so the
                # two have to be recorded side by side.
                _t0 = time.perf_counter()
                con.execute(QUERIES[f"q{qnum}"]).fetchall()
                _wall = time.perf_counter() - _t0
                print(f"[{args.condition}] exec={execution}/{args.executions} "
                      f"pos={position}/{len(qnums)} q{qnum} done wall={_wall:.3f}s", flush=True)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
