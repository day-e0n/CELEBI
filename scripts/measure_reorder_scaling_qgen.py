#!/usr/bin/env python3
"""Experiment B part 1, redone with REAL qgen-varied instances and a
parameter-aware signature.

The original measure_reorder_scaling.py repeated bare query NUMBERS -- since
cache_aware_query_reorder.query_signature(qnum, scope) only ever looks up a
static (table, column) dict keyed by qnum, two occurrences of the same
template were always "100% identical" regardless of which qgen stream (i.e.
which real substitution parameters) they'd actually come from. That's exactly
the coarse-proxy problem Part 2 (Q8 A/B/C) demonstrated with real execution --
this script fixes the *offline reorder metric* to match.

Note the reorder algorithm's scope is always "fixed_width" (see
ReorderConfig default and measure_reorder_scaling.py) -- the fixed-page cache
only ever stores fixed-width numeric/date columns, so TPC-H's classic
literal-filterable STRING dimension columns (r_name, n_name, p_type,
c_mktsegment, ...) never enter the signature at all and their substitution
values are irrelevant here. The columns that actually carry query-varying
literals *within* the fixed_width scope are date-range and numeric-threshold
filters on lineitem/orders/part, e.g. l_shipdate/l_receiptdate/o_orderdate
range bounds (with their +/- interval 'N' unit arithmetic), l_quantity and
l_discount thresholds, and p_size (= / IN-list / BETWEEN). For each
fixed-width column present in a query's base signature, this script scans the
real qgen SQL text for literal tokens following that column name and folds
them into the signature element as (table, column, literal_values) instead of
plain (table, column) -- so two instances of the same template with
different substitution parameters no longer count as full overlap on those
columns. Columns with no nearby literal (pure join keys) are left as plain
(table, column) -- consistent with the earlier finding that unfiltered joins
really do share cache regardless of instance.

Reimplements the greedy reorder (exhaustive vs --reorder-keep-first) as a
self-contained function operating on pre-computed per-position signatures,
so cache_aware_query_reorder.py (used by the actual paper experiment) is left
untouched.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from cache_aware_query_reorder import query_signature  # noqa: E402
from parse_qgen_streams import load_all_streams  # noqa: E402

NUM_STREAMS = 10

# a literal token: a date literal (optionally with +/- interval 'N' unit
# arithmetic), a quoted string, or a bare number optionally followed by a
# +/- arithmetic offset (qgen's Q6 emits discount thresholds as
# "0.06 - 0.01 and 0.06 + 0.01" rather than pre-folded bounds) -- covers
# every substitution pattern qgen actually emits for lineitem/orders/part
# fixed-width columns (shipdate/orderdate/receiptdate ranges,
# quantity/discount thresholds, p_size = / IN-list / BETWEEN).
LITERAL_TOKEN = (
    r"date\s*'[^']+'(?:\s*[+-]\s*interval\s*'[-\d]+'\s*(?:day|month|year)(?:\s*\(\d+\))?)?"
    r"|'[^']*'"
    r"|-?\d+(?:\.\d+)?(?:\s*[+-]\s*\d+(?:\.\d+)?)?"
)

# Anchored to the column as the actual LHS operand of a comparison/BETWEEN/IN
# -- NOT just "somewhere nearby" -- so pure join predicates like
# "l_orderkey = o_orderkey" (RHS is another column, not a literal-shaped
# token) never falsely pick up an unrelated literal from later in the clause.
def _column_patterns(column: str) -> list[re.Pattern]:
    col = re.escape(column)
    return [
        re.compile(rf"\b{col}\b\s*(?:not\s+)?between\s+({LITERAL_TOKEN})\s+and\s+({LITERAL_TOKEN})", re.IGNORECASE),
        re.compile(rf"\b{col}\b\s*(?:not\s+)?in\s*\(\s*([^)]*)\)", re.IGNORECASE),
        re.compile(rf"\b{col}\b\s*(?:=|<>|<=|>=|<|>)\s*({LITERAL_TOKEN})", re.IGNORECASE),
    ]


def literal_values_for_column(column: str, flat_sql: str) -> tuple[str, ...]:
    values: list[str] = []
    between_re, in_re, cmp_re = _column_patterns(column)
    for m in between_re.finditer(flat_sql):
        values.append(m.group(1).strip())
        values.append(m.group(2).strip())
    for m in in_re.finditer(flat_sql):
        values.extend(part.strip() for part in m.group(1).split(","))
    for m in cmp_re.finditer(flat_sql):
        values.append(m.group(1).strip())
    return tuple(sorted(set(values)))


def param_aware_signature(qnum: int, stream: int, sql: str, scope: str) -> frozenset:
    base = query_signature(qnum, scope)
    out = set()
    flat = " ".join(sql.split())
    for table, col in base:
        literals = literal_values_for_column(col, flat)
        if literals:
            out.add((table, col, literals))
        else:
            out.add((table, col))
    return frozenset(out)


def tiled_qnum_sequence(n: int) -> list[int]:
    return [(i % 22) + 1 for i in range(n)]


def assign_streams(qnum_sequence: list[int], num_streams: int = NUM_STREAMS) -> list[int]:
    counters: dict[int, int] = {}
    out = []
    for q in qnum_sequence:
        counters[q] = counters.get(q, 0) + 1
        out.append(((counters[q] - 1) % num_streams) + 1)
    return out


def overlap_ratio(seq_sigs: list[frozenset]) -> float:
    if len(seq_sigs) < 2:
        return 0.0
    shared = 0
    next_total = 0
    for prev_sig, next_sig in zip(seq_sigs, seq_sigs[1:]):
        shared += len(prev_sig & next_sig)
        next_total += len(next_sig)
    return shared / next_total if next_total else 0.0


def pair_rank(prev_sig: frozenset, cand_sig: frozenset, remaining_sigs: list[frozenset], cand_index: int) -> tuple[float, int, int]:
    shared = len(prev_sig & cand_sig)
    ratio = shared / len(cand_sig) if cand_sig else 0.0
    future = sum(len(cand_sig & remaining_sigs[j]) for j in range(len(remaining_sigs)) if j != cand_index)
    return ratio, shared, future


def build_greedy_path(start: int, sigs: list[frozenset]) -> list[int]:
    n = len(sigs)
    path = [start]
    remaining = [i for i in range(n) if i != start]
    while remaining:
        prev_sig = sigs[path[-1]]
        best = max(remaining, key=lambda i: pair_rank(prev_sig, sigs[i], [sigs[j] for j in remaining], remaining.index(i)))
        path.append(best)
        remaining.remove(best)
    return path


def reorder(sigs: list[frozenset], exhaustive: bool) -> tuple[list[int], float, float]:
    """Returns (best_path, elapsed_ms, overlap_ratio_after)."""
    t0 = time.perf_counter()
    n = len(sigs)
    starts = range(n) if exhaustive else [0]
    best_path = list(range(n))
    best_ratio = overlap_ratio(sigs)
    for s in starts:
        path = build_greedy_path(s, sigs)
        ratio = overlap_ratio([sigs[i] for i in path])
        if ratio > best_ratio:
            best_ratio = ratio
            best_path = path
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return best_path, elapsed_ms, best_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="20,40,60,80,100,120,140")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--scope", default="fixed_width")
    parser.add_argument("--streams-dir", type=Path, default=None, help="defaults to experiment/expB_qgen_streams")
    parser.add_argument("--num-streams", type=int, default=NUM_STREAMS)
    parser.add_argument("--out-exhaustive-csv", type=Path, required=True)
    parser.add_argument("--out-fixedstart-csv", type=Path, required=True)
    args = parser.parse_args()

    streams = load_all_streams(args.streams_dir) if args.streams_dir else load_all_streams()
    sizes = [int(x) for x in args.sizes.split(",")]

    rows_exh: list[dict[str, object]] = []
    rows_fix: list[dict[str, object]] = []
    print(f"{'N':>6}{'exh_ms':>12}{'exh_ratio':>12}{'fix_ms':>12}{'fix_ratio':>12}")
    for n in sizes:
        qnums = tiled_qnum_sequence(n)
        strms = assign_streams(qnums, args.num_streams)
        sigs = [param_aware_signature(q, s, streams[(s, q)], args.scope) for q, s in zip(qnums, strms)]

        exh_times, exh_ratios = [], []
        fix_times, fix_ratios = [], []
        for _ in range(args.trials):
            _, t_ms, ratio = reorder(sigs, exhaustive=True)
            exh_times.append(t_ms)
            exh_ratios.append(ratio)
            _, t_ms, ratio = reorder(sigs, exhaustive=False)
            fix_times.append(t_ms)
            fix_ratios.append(ratio)

        mean_exh_ms = sum(exh_times) / len(exh_times)
        mean_fix_ms = sum(fix_times) / len(fix_times)
        mean_exh_ratio = sum(exh_ratios) / len(exh_ratios)
        mean_fix_ratio = sum(fix_ratios) / len(fix_ratios)
        print(f"{n:>6}{mean_exh_ms:>12.3f}{mean_exh_ratio:>12.4f}{mean_fix_ms:>12.3f}{mean_fix_ratio:>12.4f}")
        rows_exh.append({"n": n, "mean_ms": mean_exh_ms, "overlap_ratio": mean_exh_ratio, "trials": args.trials})
        rows_fix.append({"n": n, "mean_ms": mean_fix_ms, "overlap_ratio": mean_fix_ratio, "trials": args.trials})

    for path, rows in ((args.out_exhaustive_csv, rows_exh), (args.out_fixedstart_csv, rows_fix)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
