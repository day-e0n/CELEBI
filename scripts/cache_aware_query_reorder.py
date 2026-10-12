#!/usr/bin/env python3
"""Reorder a batch of read-only analytic queries so the page cache gets more reuse.

Only the order changes; the SQL and its meaning do not. The runner calls this before handing queries
to Sirius. By default overlap counts the fixed-width columns the fixed-page cache can hold;
variable-width (string) columns are left out. The hit/overlap figures here are static estimates from
each query's column list, not GPU page-hit logs.
"""

from __future__ import annotations

import os

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
TPCH_DIR = REPO_ROOT / "test" / "tpch_performance"
sys.path.insert(0, str(TPCH_DIR))

from tpch_pin_columns import QUERY_COLUMNS  # noqa: E402

ColumnKey = tuple[str, str]

# Fixed-width columns the fixed-page cache can hold. Must match the pair-latency experiment scripts
# so their figures compare. Variable-width (string) columns are left out.
FIXED_WIDTH_COLUMNS: dict[str, set[str]] = {
    "customer": {"c_custkey", "c_nationkey", "c_acctbal"},
    "lineitem": {
        "l_orderkey",
        "l_partkey",
        "l_suppkey",
        "l_linenumber",
        "l_quantity",
        "l_extendedprice",
        "l_discount",
        "l_tax",
        "l_shipdate",
        "l_commitdate",
        "l_receiptdate",
    },
    "nation": {"n_nationkey", "n_regionkey"},
    "orders": {"o_orderkey", "o_custkey", "o_totalprice", "o_orderdate"},
    "part": {"p_partkey", "p_size", "p_retailprice"},
    "partsupp": {"ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost"},
    "region": {"r_regionkey"},
    "supplier": {"s_suppkey", "s_nationkey", "s_acctbal"},
}

SCOPE_CHOICES = ("fixed_width", "all_columns")
POLICY_CHOICES = (
    "none",
    "fixed-overlap",
    "byte-overlap",
    "byte-lru",
    "cost-ascending",
    "cost-seeded-overlap",
    "unfiltered-overlap",
    "fixed-then-variable",
    "fixed-first",
    "fixed-bytes-then-variable",
    "cacheable-first",
    "resident-overlap",
    "resident-overlap-spread",
    "big-same-adjacent",
    "big-diff-apart",
    "small-first-big-last",
    "small-first-big-runs",
    "small-first-big-ascending",
    "overlap-fit",
)

# {query: {table: [column, ...]}}, filled by load_query_table_columns().
# The union of a scan's projection AND filter columns, which is what the
# engine puts in an entry -- q6 projects 2 lineitem columns and caches 4.
QUERY_TABLE_COLUMNS: dict[str, dict[str, list[str]]] = {}

# {table: {column: bytes}}, filled by load_column_bytes(). Empty means "no size
# information", and every byte-weighted path falls back to counting columns.
COLUMN_BYTES: dict[str, dict[str, int]] = {}

# {query: {table: [filter expression, ...]}}, filled by load_query_table_filters().
# An empty list means that table was scanned WITHOUT a predicate in that query.
#
# Column overlap alone cannot tell whether a shared column is reusable. Two queries
# that both read l_shipdate overlap fully by column, and share nothing at all when
# one keeps 1995 and the other keeps 1998. Pages produced by an unfiltered scan are
# the ones that serve any later reader of those columns, so they are what a reorder
# should be clustering around. Empty here means "no filter information", and every
# scan is then treated as unfiltered -- which reproduces the column-only behaviour.
QUERY_TABLE_FILTERS: dict[str, dict[str, list[str]]] = {}

# What a shared column is worth when the query that would leave it in the cache read
# it through a predicate. Not zero: a later query with a narrower predicate over the
# same range does get served (filter_ranges_subsume in the scan manager), so filtered
# pages are worth something -- just not the full weight of a page anyone can read.
FILTERED_OVERLAP_WEIGHT = 0.25

# Which queries count as unfiltered and go in the front segment: those whose columns come from
# predicate-free scans at least this often. On TPC-H SF50, 0.5 splits the 22 queries 11/11.
UNFILTERED_PHASE_THRESHOLD = 0.5


_QUERY_AT: dict[int, int] = {}


@dataclass(frozen=True)
class ReorderConfig:
    """Reorder policy settings.

    policy decides whether and how to reorder, scope which columns overlap counts. window bounds the
    greedy candidate set; resident_column_budget sizes the diagnostic LRU.
    """

    policy: str = "fixed-overlap"
    scope: str = "fixed_width"
    window: int = 0
    keep_first: bool = False
    resident_column_budget: int = 0
    # byte-lru only: simulated cache size. The objective becomes "total bytes
    # served from cache over the whole sequence", which -- unlike adjacent-pair
    # overlap -- charges position 1 for its guaranteed miss and lets a column
    # stay resident across several unrelated queries.
    cache_budget_bytes: int = 0


@dataclass(frozen=True)
class ReorderDecision:
    """One diagnostic row per position of the reordered sequence, written to
    `query_reorder_report.csv` to explain why that query was chosen.
    """

    position: int
    selected_query: int
    original_position: int
    selected_score: float
    resident_hit_columns: int
    previous_overlap_columns: int
    future_overlap_columns: int
    query_columns: int
    resident_hit_ratio: float


@dataclass(frozen=True)
class SequenceOverlapSummary:
    """Adjacent-pair column overlap summary of a whole query sequence."""

    shared_columns: int
    next_query_columns: int
    transition_count: int
    overlap_ratio: float


@dataclass(frozen=True)
class ReorderResult:
    """Original order, reordered order and the before/after overlap."""

    original_queries: tuple[int, ...]
    reordered_queries: tuple[int, ...]
    config: ReorderConfig
    decisions: tuple[ReorderDecision, ...]
    before: SequenceOverlapSummary
    after: SequenceOverlapSummary
    elapsed_ms: float = 0.0

    @property
    def changed(self) -> bool:
        return self.original_queries != self.reordered_queries


def parse_query_list(value: str) -> list[int]:
    """Parse `1,2,6-8` or `q1,q2,q6-q8` into a list of query numbers."""

    queries: list[int] = []
    for raw in value.split(","):
        token = raw.strip().lower().removeprefix("q")
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            queries.extend(range(int(lo), int(hi) + 1))
        else:
            queries.append(int(token))
    return queries


def format_query_sequence(queries: Iterable[int]) -> str:
    """Format query numbers as `q1,q2,...`."""

    return ",".join(f"q{q}" for q in queries)


def load_column_set(module_name: str) -> str:
    """Swap QUERY_COLUMNS / FIXED_WIDTH_COLUMNS for another benchmark's column map.

    The dicts are updated in place rather than rebound: query_signature() reads the module-level
    names, so a rebind would leave importers holding the old objects.
    performance_test.load_query_set() does the same for the same reason.
    """
    import importlib

    mod = importlib.import_module(module_name)
    QUERY_COLUMNS.clear()
    QUERY_COLUMNS.update(mod.QUERY_COLUMNS)
    FIXED_WIDTH_COLUMNS.clear()
    FIXED_WIDTH_COLUMNS.update(mod.FIXED_WIDTH_COLUMNS)
    n_fixed = sum(len(v) for v in FIXED_WIDTH_COLUMNS.values())
    return f"{module_name}: {len(QUERY_COLUMNS)} queries, {n_fixed} fixed-width columns"


def query_signature(qnum: int, scope: str = "fixed_width") -> frozenset[ColumnKey]:
    """The `(table, column)` set one query needs.

    `scope=fixed_width` keeps only fixed-page cache columns, so string columns never count toward
    overlap. `scope=all_columns` is the comparison that counts every column.
    """

    if scope not in SCOPE_CHOICES:
        raise ValueError(f"bad reorder scope: {scope}")
    columns: set[ColumnKey] = set()
    for table, table_columns in QUERY_COLUMNS.get(qnum, {}).items():
        fixed = FIXED_WIDTH_COLUMNS.get(table, set())
        for column in table_columns:
            if scope == "fixed_width" and column not in fixed:
                continue
            columns.add((table, column))
    return frozenset(columns)


def worst_case_sequence(
    queries: Iterable[int], scope: str = "fixed_width"
) -> list[int]:
    """Worst arrival order: adjacent queries share as few columns as possible.

    Measuring what reordering recovers needs an order with something to recover; a benchmark's
    natural order may already be good. Greedy: pick the remaining query that overlaps least with the
    previous one, ties to the lower query number.
    """
    remaining = list(queries)
    if not remaining:
        return []
    sig = {q: query_signature(q, scope) for q in remaining}
    # Start from the query with the fewest columns. Starting from the widest one would favour the
    # cache instead: a page-cache entry accumulates columns until it freezes at its per-entry limit,
    # and the query that opens it decides its shape.
    first = min(remaining, key=lambda q: (len(sig[q]), q))
    order = [first]
    remaining.remove(first)
    while remaining:
        prev = sig[order[-1]]
        nxt = min(remaining, key=lambda q: (len(prev & sig[q]), q))
        order.append(nxt)
        remaining.remove(nxt)

    # One greedy pass is a local minimum, and not a deep one: on TPC-H it stops at
    # 19.3% adjacent overlap while a 2-opt sweep from the same start reaches 5.7%.
    # A reorder measured against the 19.3% order is measured against an arrival that
    # was already half-decent, which understates what the reorder recovers.
    def shared(seq: list[int]) -> int:
        return sum(len(sig[a] & sig[b]) for a, b in zip(seq, seq[1:]))

    best = shared(order)
    improved = True
    while improved:
        improved = False
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                order[i], order[j] = order[j], order[i]
                candidate = shared(order)
                if candidate < best:
                    best = candidate
                    improved = True
                else:
                    order[i], order[j] = order[j], order[i]
    return order


def sequence_overlap_summary(
    queries: Iterable[int], scope: str = "fixed_width"
) -> SequenceOverlapSummary:
    """How much adjacent queries share fixed-width columns: `sum(|prev & next|) / sum(|next|)`. The
    more columns two neighbours share, the more likely the second reuses pages the first left in
    VRAM.
    """

    seq = tuple(queries)
    if len(seq) < 2:
        return SequenceOverlapSummary(0, 0, 0, 0.0)
    signatures = {q: query_signature(q, scope) for q in set(seq)}
    shared = 0
    next_total = 0
    transitions = 0
    for prev_q, next_q in zip(seq, seq[1:]):
        prev_sig = signatures[prev_q]
        next_sig = signatures[next_q]
        shared += len(prev_sig & next_sig)
        next_total += len(next_sig)
        transitions += 1
    ratio = shared / next_total if next_total else 0.0
    return SequenceOverlapSummary(shared, next_total, transitions, ratio)


def worst_case_order(
    queries: Iterable[int], scope: str = "fixed_width"
) -> tuple[int, ...]:
    """Build the order that minimises overlap (the opposite of `_build_greedy_overlap_path`).

    Each step picks the candidate with the lowest overlap ratio against the previous query. Used for
    the worst-case baseline arrival order; repeats of the same template are also spread apart.
    """

    original = list(queries)
    if len(original) <= 1:
        return tuple(original)

    positions = list(range(len(original)))
    signatures = {pos: query_signature(q, scope) for pos, q in enumerate(original)}

    path = [positions[0]]
    remaining = positions[1:]
    while remaining:
        prev_sig = signatures[path[-1]]

        def rank(pos: int) -> tuple[float, int, int]:
            candidate = signatures[pos]
            shared = len(prev_sig & candidate)
            ratio = shared / len(candidate) if candidate else 0.0
            return ratio, shared, pos

        best = min(remaining, key=rank)
        path.append(best)
        remaining.remove(best)
    return tuple(original[pos] for pos in path)


def _touch_resident(
    resident_lru: list[ColumnKey],
    resident_set: set[ColumnKey],
    signature: frozenset[ColumnKey],
    budget: int,
) -> None:
    """Update the diagnostic LRU set of resident columns.

    A lightweight simulation for the reorder report (how much of what earlier queries read would
    still be resident), not the real page directory.
    """

    for column in sorted(signature):
        if column in resident_set:
            resident_lru.remove(column)
        else:
            resident_set.add(column)
        resident_lru.append(column)
    if budget > 0:
        while len(resident_lru) > budget:
            evicted = resident_lru.pop(0)
            resident_set.discard(evicted)


def _pair_overlap_rank(
    previous_position: int,
    candidate_position: int,
    remaining_positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
) -> tuple[float, int, int, int]:
    """Priority of a candidate in the greedy search: overlap ratio with the previous query, shared
    column count, future overlap with the remaining queries, then original position.
    """

    previous = signatures[previous_position]
    candidate = signatures[candidate_position]
    shared = len(previous & candidate)
    ratio = shared / len(candidate) if candidate else 0.0
    future_overlap = sum(
        len(candidate & signatures[pos])
        for pos in remaining_positions
        if pos != candidate_position
    )
    return ratio, shared, future_overlap, -candidate_position


def column_bytes(column: ColumnKey) -> int:
    """Bytes one column occupies in the cache; 1 (a column count) when no size is known."""

    table, name = column
    return COLUMN_BYTES.get(table, {}).get(name, 1)


def signature_bytes(signature: Iterable[ColumnKey]) -> int:
    return sum(column_bytes(column) for column in signature)


def load_column_bytes(path: str | Path) -> int:
    """Load the JSON probe_column_bytes.py writes into COLUMN_BYTES.

    Counting columns treats a 25-row nation.n_nationkey like lineitem.l_orderkey, so the byte-lru
    policy needs real sizes.
    """

    import json

    COLUMN_BYTES.clear()
    COLUMN_BYTES.update(json.loads(Path(path).read_text()))
    return sum(len(cols) for cols in COLUMN_BYTES.values())


def simulate_cache_hit_bytes(
    queries: Iterable[int],
    scope: str,
    budget_bytes: int,
) -> tuple[int, int]:
    """Run a byte-budget LRU and return (bytes served from cache, bytes requested).

    Unlike adjacent-pair overlap, the first query meets an empty cache, columns can survive across
    several queries, and everything is weighted by size.
    """

    resident: dict[ColumnKey, None] = {}
    resident_bytes = 0
    served = 0
    demanded = 0
    for qnum in queries:
        for column in sorted(query_signature(qnum, scope)):
            size = column_bytes(column)
            demanded += size
            if column in resident:
                served += size
                del resident[column]
            else:
                resident_bytes += size
            resident[column] = None
        if budget_bytes > 0:
            while resident_bytes > budget_bytes and resident:
                evicted, _ = next(iter(resident.items()))
                del resident[evicted]
                resident_bytes -= column_bytes(evicted)
    return served, demanded


def _build_greedy_byte_lru_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Pick next the query that would take the most bytes from a simulated byte-budget LRU.

    `_build_greedy_overlap_path` only counts shared columns with the previous query, so it cannot
    tell a 4.5 GB lineitem column from a 100 B nation column and still counts columns that were
    already evicted.
    """

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    resident: dict[ColumnKey, None] = {}
    resident_bytes = 0

    def admit(signature: frozenset[ColumnKey]) -> None:
        nonlocal resident_bytes
        for column in sorted(signature):
            if column in resident:
                del resident[column]
            else:
                resident_bytes += column_bytes(column)
            resident[column] = None
        if cfg.cache_budget_bytes > 0:
            while resident_bytes > cfg.cache_budget_bytes and resident:
                evicted, _ = next(iter(resident.items()))
                del resident[evicted]
                resident_bytes -= column_bytes(evicted)

    admit(signatures[start_position])
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        candidate_pool = remaining[:window]

        def rank(pos: int) -> tuple[int, float, int]:
            candidate = signatures[pos]
            hit = sum(column_bytes(c) for c in candidate if c in resident)
            total = signature_bytes(candidate)
            return hit, (hit / total if total else 0.0), -pos

        best_position = max(candidate_pool, key=rank)
        path.append(best_position)
        remaining.remove(best_position)
        admit(signatures[best_position])
    return path


def load_query_table_columns(path: str | Path) -> int:
    """Load the JSON probe_scan_requests.py writes (cost-ascending only)."""

    import json

    QUERY_TABLE_COLUMNS.clear()
    QUERY_TABLE_COLUMNS.update(json.loads(Path(path).read_text()))
    return len(QUERY_TABLE_COLUMNS)


def load_query_table_filters(path: str | Path) -> int:
    """Load the JSON extract_query_filters.py writes (unfiltered-overlap only)."""

    import json

    QUERY_TABLE_FILTERS.clear()
    QUERY_TABLE_FILTERS.update(json.loads(Path(path).read_text()))
    return len(QUERY_TABLE_FILTERS)


def scan_is_unfiltered(qnum: int, table: str) -> bool:
    """Whether query `qnum` reads table `table` without a predicate.

    With no filter information everything counts as unfiltered, which keeps the column-only
    behaviour when no JSON is given.
    """

    if not QUERY_TABLE_FILTERS:
        return True
    per_table = QUERY_TABLE_FILTERS.get(f"q{qnum}")
    if per_table is None or table not in per_table:
        return True
    return not per_table[table]


def unfiltered_signature(qnum: int, scope: str = "fixed_width") -> frozenset[ColumnKey]:
    """The part of `query_signature` that comes from unfiltered scans."""

    return frozenset(
        (table, column)
        for table, column in query_signature(qnum, scope)
        if scan_is_unfiltered(qnum, table)
    )


def estimated_scan_bytes(qnum: int) -> int:
    """Estimated decoded bytes a query reads. Not a cost model, but it correlates with measured scan
    time and needs only the plan.
    """

    per_table = QUERY_TABLE_COLUMNS.get(f"q{qnum}", {})
    return sum(
        column_bytes((table, column))
        for table, columns in per_table.items()
        for column in columns
    )


def _cost_seeded_overlap_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Start from the cheapest query, then continue with overlap greedy.

    `fixed-overlap` picks the start that maximises the overall overlap ratio, whose denominator
    never includes the first query, so the first slot looks free and the most expensive scan can
    land there -- the one slot where the cache is guaranteed empty. Here only that slot is fixed by
    cost: spend the cold start on the cheapest scan and let overlap order the rest.
    """

    del start_position
    seed = min(positions, key=lambda pos: (estimated_scan_bytes(_QUERY_AT[pos]), pos))
    return _build_greedy_overlap_path(seed, positions, signatures, cfg)


def _cost_ascending_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Ascending scan cost, ignoring adjacency.

    With per-entry caching, order matters by accumulation rather than adjacency: an entry gathers
    columns until it freezes at its per-entry limit and nothing widens it afterwards. Expensive
    scans should therefore arrive after cheap ones have widened the entry.
    """

    del start_position, signatures, cfg
    return sorted(
        positions, key=lambda pos: (estimated_scan_bytes(_QUERY_AT[pos]), pos)
    )


def _byte_pair_overlap_rank(
    previous_position: int,
    candidate_position: int,
    remaining_positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
) -> tuple[float, int, int, int]:
    """Like `_pair_overlap_rank`, but measures shared columns in bytes rather than count.

    Unlike `byte-lru` it does not simulate the cache: a residency model's errors would flow straight
    into the order, so only the observable bytes shared with the previous query are used.
    """

    previous = signatures[previous_position]
    candidate = signatures[candidate_position]
    shared = sum(column_bytes(c) for c in previous & candidate)
    total = signature_bytes(candidate)
    ratio = shared / total if total else 0.0
    future = sum(
        sum(column_bytes(c) for c in candidate & signatures[pos])
        for pos in remaining_positions
        if pos != candidate_position
    )
    return ratio, shared, future, -candidate_position


def _build_greedy_byte_overlap_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Chain queries by largest byte overlap with the previous one."""

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        candidate_pool = remaining[:window]
        previous_position = path[-1]
        best_position = max(
            candidate_pool,
            key=lambda pos: _byte_pair_overlap_rank(
                previous_position, pos, remaining, signatures
            ),
        )
        path.append(best_position)
        remaining.remove(best_position)
    return path


def _fixed_then_variable_pair_overlap_rank(
    previous_position: int,
    candidate_position: int,
    remaining_positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
) -> tuple[float, int, float, int, int, int]:
    """Rank by fixed-width overlap first and variable-width overlap second.

    `fixed-overlap` never looks at strings, so where one string column outweighs everything else
    whether two string queries are adjacent never enters the order; `byte-overlap` lets strings
    dominate every decision. Here strings only break ties in fixed-width overlap.
    """

    prev_fixed = signatures[previous_position]
    cand_fixed = signatures[candidate_position]
    fixed_shared = len(prev_fixed & cand_fixed)
    fixed_ratio = fixed_shared / len(cand_fixed) if cand_fixed else 0.0

    prev_var = _variable_signature(previous_position)
    cand_var = _variable_signature(candidate_position)
    var_shared = sum(column_bytes(c) for c in prev_var & cand_var)
    var_total = sum(column_bytes(c) for c in cand_var)
    var_ratio = var_shared / var_total if var_total else 0.0

    future = sum(
        len(cand_fixed & signatures[pos])
        for pos in remaining_positions
        if pos != candidate_position
    )
    return fixed_ratio, fixed_shared, var_ratio, var_shared, future, -candidate_position


def _variable_signature(position: int) -> frozenset[ColumnKey]:
    """Variable-width (string) columns the query at this position reads."""

    qnum = _QUERY_AT[position] if _QUERY_AT else position
    return query_signature(qnum, "all_columns") - query_signature(qnum, "fixed_width")


def _build_fixed_then_variable_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Chain queries; on equal fixed-width overlap, prefer the larger string overlap."""

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        candidate_pool = remaining[:window]
        previous_position = path[-1]
        best_position = max(
            candidate_pool,
            key=lambda pos: _fixed_then_variable_pair_overlap_rank(
                previous_position, pos, remaining, signatures
            ),
        )
        path.append(best_position)
        remaining.remove(best_position)
    return path


def _unfiltered_pair_overlap_rank(
    previous_position: int,
    candidate_position: int,
    remaining_positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
) -> tuple[float, float, float, int]:
    """Rank the next query preferring columns left by unfiltered scans.

    `_pair_overlap_rank` counts every shared column as 1, but a column the previous query read
    without a predicate serves anyone, while one read under a predicate serves only narrower
    predicates. So each shared column is weighted by whether the previous (the query that leaves it
    in the cache) scanned it unfiltered.
    """

    previous_query = _QUERY_AT[previous_position]
    candidate = signatures[candidate_position]
    shared = signatures[previous_position] & candidate

    def weight(column: ColumnKey) -> float:
        # The previous query leaves the column in the cache; an unfiltered scan there scores a full
        # point.
        return (
            1.0
            if scan_is_unfiltered(previous_query, column[0])
            else FILTERED_OVERLAP_WEIGHT
        )

    weighted = sum(weight(column) for column in shared)
    ratio = weighted / len(candidate) if candidate else 0.0
    # Tie-break: prefer candidates that will leave more unfiltered columns behind.
    candidate_query = _QUERY_AT[candidate_position]
    future = sum(
        len(unfiltered_signature(candidate_query) & signatures[pos])
        for pos in remaining_positions
        if pos != candidate_position
    )
    return ratio, weighted, float(future), -candidate_position


def unfiltered_policy_applies(queries: Iterable[int]) -> bool:
    """Whether unfiltered-first reordering has anything to separate on this workload.

    The policy rests on unfiltered pages being usable by anyone. Once the scan-size gate opens,
    scans carrying a join's dynamic filter also cache pre-filter data with no predicate attached. If
    no query reads the dominant table without a join, every page of that table is predicate-free and
    the policy only shuffles the order. One such query is enough to fill the cache with
    predicate-free pages, so the test is existence, not a ratio. Without filter information, return
    True and keep the existing behaviour.
    """

    if not QUERY_TABLE_FILTERS:
        return True
    per_query = {q: QUERY_TABLE_FILTERS.get(f"q{q}", {}) for q in queries}
    counts: dict[str, int] = {}
    for tables in per_query.values():
        for table in tables:
            counts[table] = counts.get(table, 0) + 1
    if not counts:
        return True
    dominant = max(counts, key=lambda t: (counts[t], t))
    return any(len(tables) == 1 and dominant in tables for tables in per_query.values())


def unfiltered_fraction(qnum: int, scope: str = "fixed_width") -> float:
    """Fraction of a query's columns that come from predicate-free scans."""

    signature = query_signature(qnum, scope)
    if not signature:
        return 0.0
    return len(unfiltered_signature(qnum, scope)) / len(signature)


def _greedy_within(
    start_position: int,
    pool: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Walk the pool from `start_position` with the unfiltered-weighted greedy."""

    path = [start_position]
    remaining = [pos for pos in pool if pos != start_position]
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        previous_position = path[-1]
        best_position = max(
            remaining[:window],
            key=lambda pos: _unfiltered_pair_overlap_rank(
                previous_position, pos, remaining, signatures
            ),
        )
        path.append(best_position)
        remaining.remove(best_position)
    return path


def _fixed_share(position: int) -> float:
    """Fraction of the bytes the query at this position reads that are fixed-width."""

    qnum = _QUERY_AT[position] if _QUERY_AT else position
    fixed = query_signature(qnum, "fixed_width")
    every = query_signature(qnum, "all_columns")
    total = sum(column_bytes(c) for c in every)
    if total <= 0:
        # No size information: fall back to column counts.
        return len(fixed) / len(every) if every else 0.0
    return sum(column_bytes(c) for c in fixed) / total


FIXED_PHASE_THRESHOLD = 0.5


def _build_fixed_first_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Put fixed-width-dominated queries in a front segment and string-dominated ones after.

    The `fixed-overlap` family ranks every step by fixed-width overlap, so string queries still slip
    in between, take the cache first and leave fixed-width columns no room. Split into segments
    instead of ranks: while the cache is empty the fixed-width queries settle in and reuse each
    other, then the string queries run.
    """

    front = [pos for pos in positions if _fixed_share(pos) >= FIXED_PHASE_THRESHOLD]
    back = [pos for pos in positions if pos not in set(front)]
    if not front or not back:
        return _build_greedy_overlap_path(start_position, positions, signatures, cfg)
    if start_position not in front:
        front = [start_position] + front
        back = [pos for pos in back if pos != start_position]
    path = _greedy_within(start_position, front, signatures, cfg)
    bridge = max(
        back, key=lambda pos: _pair_overlap_rank(path[-1], pos, back, signatures)
    )
    path.extend(_greedy_within(bridge, back, signatures, cfg))
    return path


def _fixed_bytes_then_variable_rank(
    previous_position: int,
    candidate_position: int,
    remaining_positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
) -> tuple[int, float, float, float, int]:
    """Rank by fixed-width overlap bytes, then by the smallest string overlap; once no fixed-width
    overlap is left, by string overlap.

    What actually gets reused is a column whose row groups all fit the budget and that several
    queries read, which is almost always fixed-width. Adjacent string queries rarely help: their
    pages are evicted before reuse and sweep out fixed-width pages on the way.
    """

    prev_fixed = signatures[previous_position]
    cand_fixed = signatures[candidate_position]
    fixed_shared = sum(column_bytes(c) for c in prev_fixed & cand_fixed)

    prev_var = _variable_signature(previous_position)
    cand_var = _variable_signature(candidate_position)
    var_shared = sum(column_bytes(c) for c in prev_var & cand_var)

    future = sum(
        sum(column_bytes(c) for c in cand_fixed & signatures[pos])
        for pos in remaining_positions
        if pos != candidate_position
    )
    if fixed_shared > 0:
        # First fixed-width bytes, then the smaller string overlap (negated so both maximise).
        return 1, fixed_shared, -var_shared, future, -candidate_position
    # Nothing left to chain on fixed-width columns: chain on string overlap.
    return 0, 0.0, var_shared, future, -candidate_position


def _build_fixed_bytes_then_variable_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Chain queries with `_fixed_bytes_then_variable_rank`."""

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        candidate_pool = remaining[:window]
        previous_position = path[-1]
        best_position = max(
            candidate_pool,
            key=lambda pos: _fixed_bytes_then_variable_rank(
                previous_position, pos, remaining, signatures
            ),
        )
        path.append(best_position)
        remaining.remove(best_position)
    return path


#: Fraction of the budget above which a column counts as too big to be wholly resident. At 1.0 a
#: column must exceed the whole budget. Lower values call a column big once it takes part of the
#: budget, since the rest then compete for what is left; where that line sits may differ per
#: workload. Set with SIRIUS_REORDER_BIG_COLUMN_RATIO.
BIG_COLUMN_BUDGET_RATIO = float(
    os.environ.get("SIRIUS_REORDER_BIG_COLUMN_RATIO", "1.0")
)


def _column_never_resident(column: ColumnKey, cache_budget_bytes: int) -> bool:
    """Whether this column can never be wholly resident within the budget."""

    if cache_budget_bytes <= 0:
        return False
    return column_bytes(column) > cache_budget_bytes * BIG_COLUMN_BUDGET_RATIO


def _query_is_cacheable(position: int, cfg: ReorderConfig) -> bool:
    """Whether every column this query reads fits within the budget."""

    qnum = _QUERY_AT[position] if _QUERY_AT else position
    return not any(
        _column_never_resident(c, cfg.cache_budget_bytes)
        for c in query_signature(qnum, "all_columns")
    )


def _build_cacheable_first_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Put queries whose columns all fit the budget first and the others last.

    Same criterion as eviction priority. When a column is larger than the budget its row groups
    cannot all be resident at once, so the row group a later query wants is already gone. Pulling
    such queries forward by overlap creates no reuse and only sweeps out pages that could have
    stayed resident. So they go last, and the front segment is chained by byte overlap among queries
    that can actually reuse each other.
    """

    front = [pos for pos in positions if _query_is_cacheable(pos, cfg)]
    back = [pos for pos in positions if pos not in set(front)]
    if not front or not back:
        return _build_greedy_byte_overlap_path(
            start_position, positions, signatures, cfg
        )
    if start_position not in front:
        front = [start_position] + front
        back = [pos for pos in back if pos != start_position]
    path = _greedy_within(start_position, front, signatures, cfg)
    bridge = max(
        back, key=lambda pos: _byte_pair_overlap_rank(path[-1], pos, back, signatures)
    )
    path.extend(_greedy_within(bridge, back, signatures, cfg))
    return path


def _resident_signature(position: int, cache_budget_bytes: int) -> frozenset[ColumnKey]:
    """Columns the query at this position reads that fit wholly within the budget."""

    qnum = _QUERY_AT[position] if _QUERY_AT else position
    return frozenset(
        c
        for c in query_signature(qnum, "all_columns")
        if not _column_never_resident(c, cache_budget_bytes)
    )


def _resident_pair_overlap_rank(
    previous_position: int,
    candidate_position: int,
    remaining_positions: list[int],
    cache_budget_bytes: int,
) -> tuple[float, int, int, int]:
    """Rank by overlap bytes over the columns that fit the budget.

    A column larger than the budget is never served from the cache: the row group the next query
    wants has already been evicted. Counting its overlap glues those queries together, serves
    nothing, and makes the cache re-insert the same column repeatedly, so it is left out of the
    overlap.
    """

    prev = _resident_signature(previous_position, cache_budget_bytes)
    cand = _resident_signature(candidate_position, cache_budget_bytes)
    shared = sum(column_bytes(c) for c in prev & cand)
    total = sum(column_bytes(c) for c in cand)
    ratio = shared / total if total else 0.0
    future = sum(
        sum(
            column_bytes(c) for c in cand & _resident_signature(pos, cache_budget_bytes)
        )
        for pos in remaining_positions
        if pos != candidate_position
    )
    return ratio, shared, future, -candidate_position


def _build_resident_overlap_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Chain queries by overlap over columns that fit the budget only."""

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        candidate_pool = remaining[:window]
        previous_position = path[-1]
        best_position = max(
            candidate_pool,
            key=lambda pos: _resident_pair_overlap_rank(
                previous_position, pos, remaining, cfg.cache_budget_bytes
            ),
        )
        path.append(best_position)
        remaining.remove(best_position)
    return path


def _reads_oversized_column(position: int, cache_budget_bytes: int) -> bool:
    """Whether this query reads a column larger than the budget."""

    qnum = _QUERY_AT[position] if _QUERY_AT else position
    return any(
        _column_never_resident(c, cache_budget_bytes)
        for c in query_signature(qnum, "all_columns")
    )


def _build_resident_overlap_spread_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Chain by overlap over columns that fit the budget, and keep queries that read an oversized
    column apart.

    Oversized columns are left out of the overlap (`resident-overlap`), and queries reading them are
    spaced apart so they do not repeatedly evict and re-insert each other. The spacing target is the
    total query count divided by the number of such queries, i.e. an even spread.
    """

    budget = cfg.cache_budget_bytes
    oversized = {pos for pos in positions if _reads_oversized_column(pos, budget)}
    if not oversized or len(oversized) == len(positions):
        # Nothing is too big to be resident, so there is nothing to push back and no
        # reason to disturb the caller's order. Falling through to another policy is
        # what cost TPC-H 6.8%: no TPC-H query reads a column over the budget, so the
        # measurement was of resident-overlap, which is worse than not reordering.
        return list(positions)
    spacing = max(1, len(positions) // len(oversized))

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    since_oversized = 0 if start_position in oversized else spacing
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        pool = remaining[:window]
        # Until the spacing is met, drop oversized-column queries from the candidates; use them
        # anyway if nothing else is left.
        allowed = (
            [pos for pos in pool if pos not in oversized]
            if since_oversized < spacing
            else pool
        )
        if not allowed:
            allowed = pool
        previous_position = path[-1]
        best_position = max(
            allowed,
            key=lambda pos: _resident_pair_overlap_rank(
                previous_position, pos, remaining, budget
            ),
        )
        path.append(best_position)
        remaining.remove(best_position)
        since_oversized = 0 if best_position in oversized else since_oversized + 1
    return path


def _oversized_signature(
    position: int, cache_budget_bytes: int
) -> frozenset[ColumnKey]:
    """Columns the query at this position reads that exceed the budget."""

    qnum = _QUERY_AT[position] if _QUERY_AT else position
    return frozenset(
        c
        for c in query_signature(qnum, "all_columns")
        if _column_never_resident(c, cache_budget_bytes)
    )


def _build_big_same_adjacent_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Keep queries that read the same oversized column together and different oversized columns
    apart.

    Spacing every oversized-column query apart made things worse. The pairs worth separating are
    those reading DIFFERENT oversized columns, which evict each other wholesale when they alternate
    under one budget; a pair reading the same column has nothing to evict. So the same oversized
    column is chained directly, and different ones get queries that fit the budget in between.
    """

    budget = cfg.cache_budget_bytes
    big_of = {pos: _oversized_signature(pos, budget) for pos in positions}
    if not any(big_of.values()):
        # See _small_first_big_tail: with nothing oversized, leave the order alone.
        return list(positions)

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        pool = remaining[:window]
        previous_position = path[-1]
        prev_big = big_of[previous_position]

        def tier(pos: int) -> int:
            cand_big = big_of[pos]
            if prev_big and cand_big == prev_big:
                return 2  # same oversized column: chain directly
            if not cand_big:
                return 1  # no oversized column: a good buffer
            return 0  # different oversized column: last

        best_position = max(
            pool,
            key=lambda pos: (tier(pos),)
            + _resident_pair_overlap_rank(previous_position, pos, remaining, budget),
        )
        path.append(best_position)
        remaining.remove(best_position)
    return path


def _build_big_diff_apart_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Chain by overlap over columns that fit the budget, but defer a DIFFERENT oversized column
    than the previous one until the spacing is met.

    The same observation as `big-same-adjacent` with the opposite force: that policy pulls the same
    oversized column closer, this one pushes different ones away. The same column needs no spacing,
    so a run of queries reading it stays together.
    """

    budget = cfg.cache_budget_bytes
    big_of = {pos: _oversized_signature(pos, budget) for pos in positions}
    n_big = sum(1 for v in big_of.values() if v)
    if n_big == 0 or n_big == len(positions):
        # Nothing is too big to be resident, so there is nothing to push back and no
        # reason to disturb the caller's order. Falling through to another policy is
        # what cost TPC-H 6.8%: no TPC-H query reads a column over the budget, so the
        # measurement was of resident-overlap, which is worse than not reordering.
        return list(positions)
    spacing = max(
        1,
        len(positions) // max(1, len(set(frozenset(v) for v in big_of.values() if v))),
    )

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    last_big = big_of[start_position]
    since_switch = 0
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        pool = remaining[:window]
        previous_position = path[-1]

        def allowed(pos: int) -> bool:
            cand_big = big_of[pos]
            if not cand_big or not last_big or cand_big == last_big:
                return True  # no oversized column, or the same as before: any time
            return since_switch >= spacing

        pool_ok = [pos for pos in pool if allowed(pos)] or pool
        best_position = max(
            pool_ok,
            key=lambda pos: _resident_pair_overlap_rank(
                previous_position, pos, remaining, budget
            ),
        )
        path.append(best_position)
        remaining.remove(best_position)
        cand_big = big_of[best_position]
        if cand_big and cand_big != last_big:
            last_big, since_switch = cand_big, 0
        else:
            since_switch += 1
    return path


def _build_overlap_fit_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Overlap first, then whether the overlapping columns fit the budget.

    Lexicographic rank of the next query: shares any column with the previous query > the shared
    columns' bytes fit the cache budget > overlap byte ratio > overlap bytes > overlap bytes with
    the remaining queries. Unlike `small-first-big-last`, queries reading oversized columns are not
    sent to the back explicitly; they fall back on the second criterion.
    """

    budget = cfg.cache_budget_bytes

    def all_columns(pos: int) -> frozenset[ColumnKey]:
        qnum = _QUERY_AT[pos] if _QUERY_AT else pos
        return query_signature(qnum, "all_columns")

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    while remaining:
        previous = all_columns(path[-1])

        def key(pos: int):
            cand = all_columns(pos)
            shared = previous & cand
            shared_bytes = sum(column_bytes(c) for c in shared)
            total = sum(column_bytes(c) for c in cand)
            fits = budget <= 0 or shared_bytes <= budget
            future = sum(
                sum(column_bytes(c) for c in cand & all_columns(other))
                for other in remaining
                if other != pos
            )
            return (
                1 if shared else 0,
                1 if fits else 0,
                shared_bytes / total if total else 0.0,
                shared_bytes,
                future,
                -pos,
            )

        best_position = max(remaining, key=key)
        path.append(best_position)
        remaining.remove(best_position)
    return path


def _build_small_first_big_last_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Send queries that read oversized columns as far back as possible, chaining the front by
    small-column overlap only.

    Tier 1: queries reading no oversized column, by small-column bytes shared with the previous
    query. Tier 0: queries reading an oversized column, the same oversized column as the previous
    query first, so different oversized columns do not alternate at the back.

    Unlike `big-same-adjacent`, an oversized-column query is always pushed back rather than chained
    right after another one.
    """

    budget = cfg.cache_budget_bytes
    big_of = {pos: _oversized_signature(pos, budget) for pos in positions}
    if not any(big_of.values()):
        # Nothing is too big to be resident, so there is nothing to push back and no
        # reason to disturb the caller's order. Falling through to another policy is
        # what cost TPC-H 6.8%: no TPC-H query reads a column over the budget, so the
        # measurement was of resident-overlap, which is worse than not reordering.
        return list(positions)

    # Do not start on a query that reads one of the oversized columns: the caller's first
    # query is used as the seed, and if that one reads URL the whole point -- pushing those
    # to the back -- is lost at position 1.
    if big_of[start_position]:
        small = [pos for pos in positions if not big_of[pos]]
        if small:
            start_position = small[0]

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        pool = remaining[:window]
        previous_position = path[-1]
        prev_big = big_of[previous_position]

        def key(pos: int):
            cand_big = big_of[pos]
            if not cand_big:
                # Tier 1: by small-column overlap bytes.
                return (1,) + _resident_pair_overlap_rank(
                    previous_position, pos, remaining, budget
                )
            # Tier 0: the same oversized column as the previous query first.
            same = 1 if (prev_big and cand_big == prev_big) else 0
            shared_big = sum(column_bytes(c) for c in prev_big & cand_big)
            return (0, same, shared_big, 0.0, 0, -pos)

        best_position = max(pool, key=key)
        path.append(best_position)
        remaining.remove(best_position)
    return path


def _small_first_big_tail(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
    tail_key,
) -> list[int]:
    """Shared skeleton of `small-first-big-last`; `tail_key` swaps only the back segment's ordering."""

    budget = cfg.cache_budget_bytes
    big_of = {pos: _oversized_signature(pos, budget) for pos in positions}
    if not any(big_of.values()):
        # Nothing here is too big to be resident, so there is nothing to push to the back and
        # no reason to disturb the order the caller gave. Falling through to a different
        # policy is what made this cost TPC-H 6.8%: no TPC-H query reads a column over the
        # budget (the largest one any of them reads is l_shipinstruct at 4.47 GB), so the
        # measurement was of resident-overlap, which is worse than not reordering.
        return list(positions)
    if big_of[start_position]:
        small = [pos for pos in positions if not big_of[pos]]
        if small:
            start_position = small[0]

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        pool = remaining[:window]
        previous_position = path[-1]
        prev_big = big_of[previous_position]

        def key(pos: int):
            if not big_of[pos]:
                return (1,) + _resident_pair_overlap_rank(
                    previous_position, pos, remaining, budget
                )
            return (0,) + tail_key(pos, prev_big, big_of)

        best_position = max(pool, key=key)
        path.append(best_position)
        remaining.remove(best_position)
    return path


def _build_small_first_big_runs_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Order the back segment so each oversized column appears in one contiguous run.

    An oversized column cannot be wholly resident, so if it appears twice it is read in full again
    in between. Prefer the candidate sharing the most oversized columns with the previous query, and
    among those the one introducing the fewest new oversized columns.
    """

    def tail_key(pos, prev_big, big_of):
        cand = big_of[pos]
        shared = sum(column_bytes(c) for c in prev_big & cand)
        newly = len(cand - prev_big)
        return (shared, -newly, -pos)

    return _small_first_big_tail(start_position, positions, signatures, cfg, tail_key)


def _build_small_first_big_ascending_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Order the back segment from the smallest oversized column to the largest.

    A column bigger than the whole budget flushes the cache whatever is done, so that flush is best
    left to the very end of the run, where nothing is left to sweep out.
    """

    def tail_key(pos, prev_big, big_of):
        cand = big_of[pos]
        biggest = max((column_bytes(c) for c in cand), default=0)
        shared = sum(column_bytes(c) for c in prev_big & cand)
        return (-biggest, shared, -pos)

    return _small_first_big_tail(start_position, positions, signatures, cfg, tail_key)


def _build_unfiltered_first_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Put predicate-free queries in a front segment and filtered ones after.

    A column-overlap greedy counts `l_shipdate < 1995` and `l_shipdate >= 1998` as a perfect pair,
    yet neither can use a single row of the other's pages. Only pages read without a predicate serve
    anyone, so run those queries first to fill the cache with generally usable content, then the
    narrowly filtered ones.
    """

    front = [
        pos
        for pos in positions
        if unfiltered_fraction(_QUERY_AT[pos], cfg.scope) >= UNFILTERED_PHASE_THRESHOLD
    ]
    back = [pos for pos in positions if pos not in set(front)]
    if start_position not in front:
        # The reorder restricts the start to unfiltered queries; if a filtered one arrives anyway,
        # keep it in the front so the policy itself still holds.
        front = [start_position] + front
        back = [pos for pos in back if pos != start_position]
    path = _greedy_within(start_position, front, signatures, cfg)
    if back:
        bridge = max(
            back,
            key=lambda pos: _unfiltered_pair_overlap_rank(
                path[-1], pos, back, signatures
            ),
        )
        path.extend(_greedy_within(bridge, back, signatures, cfg))
    return path


def sequence_unfiltered_overlap(
    queries: Iterable[int], scope: str = "fixed_width"
) -> float:
    """Weighted sum of shared columns over adjacent pairs: the unfiltered-overlap objective."""

    sequence = list(queries)
    total = 0.0
    for previous, nxt in zip(sequence, sequence[1:]):
        shared = query_signature(previous, scope) & query_signature(nxt, scope)
        total += sum(
            1.0 if scan_is_unfiltered(previous, table) else FILTERED_OVERLAP_WEIGHT
            for table, _ in shared
        )
    return total


def _build_greedy_overlap_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Greedy path from one start position, picking the next query with the largest overlap."""

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        candidate_pool = remaining[:window]
        previous_position = path[-1]
        best_position = max(
            candidate_pool,
            key=lambda pos: _pair_overlap_rank(
                previous_position, pos, remaining, signatures
            ),
        )
        path.append(best_position)
        remaining.remove(best_position)
    return path


def _resident_aware_rank(
    candidate_position: int,
    previous_position: int,
    remaining_positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    resident_set: set[ColumnKey],
) -> tuple[float, int, int, int, int]:
    """Priority of the next candidate in the budget-aware greedy.

    Overlap with what is actually resident in the simulated cache comes first: a column already
    evicted under a tight budget is useless however much it overlaps the previous query.
    """

    candidate = signatures[candidate_position]
    previous = signatures[previous_position]
    resident_hits = len(candidate & resident_set)
    resident_ratio = resident_hits / len(candidate) if candidate else 0.0
    previous_overlap = len(previous & candidate)
    future_overlap = sum(
        len(candidate & signatures[pos])
        for pos in remaining_positions
        if pos != candidate_position
    )
    return (
        resident_ratio,
        resident_hits,
        previous_overlap,
        future_overlap,
        -candidate_position,
    )


def _build_greedy_resident_aware_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """Build the order while simulating a capacity-limited LRU cache.

    `_build_greedy_overlap_path` only looks at overlap with the previous query. This version updates
    residency with `_touch_resident` at every step and prefers the query that overlaps most with
    what is really resident.
    """

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    resident_lru: list[ColumnKey] = []
    resident_set: set[ColumnKey] = set()
    _touch_resident(
        resident_lru,
        resident_set,
        signatures[start_position],
        cfg.resident_column_budget,
    )

    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        candidate_pool = remaining[:window]
        previous_position = path[-1]
        best_position = max(
            candidate_pool,
            key=lambda pos: _resident_aware_rank(
                pos, previous_position, remaining, signatures, resident_set
            ),
        )
        path.append(best_position)
        remaining.remove(best_position)
        _touch_resident(
            resident_lru,
            resident_set,
            signatures[best_position],
            cfg.resident_column_budget,
        )
    return path


def _decision_score(
    original_position: int,
    signature: frozenset[ColumnKey],
    pending_positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    resident_set: set[ColumnKey],
    previous_signature: frozenset[ColumnKey],
) -> tuple[float, int, int, int, int, float]:
    """Score and hit diagnostics for the CSV report.

    The reorder itself decides on `_pair_overlap_rank` and the overall overlap summary; this score
    only puts resident hits, previous-query overlap and future overlap side by side for a reader.
    """

    resident_hits = len(signature & resident_set)
    previous_hits = len(signature & previous_signature)
    future_overlap = sum(
        len(signature & signatures[pos])
        for pos in pending_positions
        if pos != original_position
    )
    footprint = len(signature)
    misses = max(footprint - resident_hits, 0)
    score = (
        4.0 * resident_hits
        + 2.0 * previous_hits
        + 0.25 * future_overlap
        - 0.10 * misses
    )
    hit_ratio = resident_hits / footprint if footprint else 0.0
    return score, resident_hits, previous_hits, future_overlap, footprint, hit_ratio


def reorder_query_sequence(
    queries: Iterable[int], config: ReorderConfig | None = None
) -> ReorderResult:
    """Reorder a query sequence into a cache-aware order.

    Build a greedy path from each possible start and keep the order with the largest overall
    fixed-column overlap. If that does not improve on the given order, keep the given order.
    """

    start_time = time.perf_counter()
    cfg = config or ReorderConfig()
    original = tuple(int(q) for q in queries)
    before = sequence_overlap_summary(original, cfg.scope)

    def finish(
        reordered: tuple[int, ...],
        decisions: tuple[ReorderDecision, ...],
        after: SequenceOverlapSummary,
    ) -> ReorderResult:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return ReorderResult(
            original, reordered, cfg, decisions, before, after, elapsed_ms
        )

    if cfg.policy == "none" or len(original) <= 1:
        after = sequence_overlap_summary(original, cfg.scope)
        return finish(original, tuple(), after)
    if cfg.policy not in POLICY_CHOICES:
        raise ValueError(f"bad reorder policy: {cfg.policy}")

    queries_by_position = dict(enumerate(original, start=1))
    positions = list(queries_by_position)
    signatures = {
        original_position: query_signature(qnum, cfg.scope)
        for original_position, qnum in queries_by_position.items()
    }

    if cfg.policy == "unfiltered-overlap" and not unfiltered_policy_applies(original):
        # On this workload a predicate's presence does not separate reusability; leave the order
        # alone.
        return finish(original, tuple(), before)
    if cfg.policy == "unfiltered-overlap" and not cfg.keep_first:
        # The first query decides what the cache holds first. Opening with a filtered scan leaves
        # pages most later queries cannot read, so only queries with unfiltered scans are tried as
        # the start, widest unfiltered columns first.
        seeds = [
            pos
            for pos in positions
            if unfiltered_fraction(queries_by_position[pos], cfg.scope)
            >= UNFILTERED_PHASE_THRESHOLD
        ]
        start_positions = sorted(
            seeds or positions,
            key=lambda pos: (
                -unfiltered_fraction(queries_by_position[pos], cfg.scope),
                -len(query_signature(queries_by_position[pos], cfg.scope)),
                pos,
            ),
        )
    elif cfg.keep_first:
        start_positions = positions[:1]
    elif cfg.window > 0:
        start_positions = positions[: min(cfg.window, len(positions))]
    else:
        start_positions = positions

    best_path = positions
    best_queries = original
    best_summary = before
    best_rank = (before.overlap_ratio, before.shared_columns, -positions[0])
    global _QUERY_AT
    _QUERY_AT = queries_by_position
    if cfg.policy == "cost-seeded-overlap":
        path = _cost_seeded_overlap_path(positions[0], positions, signatures, cfg)
        reordered = tuple(queries_by_position[pos] for pos in path)
        return finish(
            reordered, tuple(), sequence_overlap_summary(reordered, cfg.scope)
        )
    if cfg.policy == "cost-ascending":
        # No start search: one sort decides the order, so skip the best-of-start-positions loop and
        # the overlap safeguard below.
        path = _cost_ascending_path(positions[0], positions, signatures, cfg)
        reordered = tuple(queries_by_position[pos] for pos in path)
        return finish(
            reordered, tuple(), sequence_overlap_summary(reordered, cfg.scope)
        )
    if cfg.policy == "unfiltered-overlap":
        path_builders = [_build_unfiltered_first_path]
    elif cfg.policy == "byte-lru":
        path_builders = [_build_greedy_byte_lru_path]
    elif cfg.policy == "byte-overlap":
        path_builders = [_build_greedy_byte_overlap_path]
    elif cfg.policy == "fixed-then-variable":
        path_builders = [_build_fixed_then_variable_path]
    elif cfg.policy == "fixed-first":
        path_builders = [_build_fixed_first_path]
    elif cfg.policy == "fixed-bytes-then-variable":
        path_builders = [_build_fixed_bytes_then_variable_path]
    elif cfg.policy == "cacheable-first":
        path_builders = [_build_cacheable_first_path]
    elif cfg.policy == "resident-overlap":
        path_builders = [_build_resident_overlap_path]
    elif cfg.policy == "resident-overlap-spread":
        path_builders = [_build_resident_overlap_spread_path]
    elif cfg.policy == "big-same-adjacent":
        path_builders = [_build_big_same_adjacent_path]
    elif cfg.policy == "big-diff-apart":
        path_builders = [_build_big_diff_apart_path]
    elif cfg.policy == "small-first-big-last":
        path_builders = [_build_small_first_big_last_path]
    elif cfg.policy == "overlap-fit":
        path_builders = [_build_overlap_fit_path]
    elif cfg.policy == "small-first-big-runs":
        path_builders = [_build_small_first_big_runs_path]
    elif cfg.policy == "small-first-big-ascending":
        path_builders = [_build_small_first_big_ascending_path]
    elif cfg.resident_column_budget > 0:
        path_builders = [_build_greedy_resident_aware_path]
    else:
        path_builders = [_build_greedy_overlap_path]

    def score(
        candidate_queries: tuple[int, ...], summary: SequenceOverlapSummary, start: int
    ):
        if cfg.policy == "byte-overlap":
            shared = sum(
                sum(
                    column_bytes(c)
                    for c in query_signature(a, cfg.scope)
                    & query_signature(b, cfg.scope)
                )
                for a, b in zip(candidate_queries, candidate_queries[1:])
            )
            return (shared, summary.shared_columns, -start)
        if cfg.policy == "unfiltered-overlap":
            return (
                sequence_unfiltered_overlap(candidate_queries, cfg.scope),
                summary.shared_columns,
                -start,
            )
        if cfg.policy != "byte-lru":
            return (summary.overlap_ratio, summary.shared_columns, -start)
        served, _ = simulate_cache_hit_bytes(
            candidate_queries, cfg.scope, cfg.cache_budget_bytes
        )
        return (served, summary.shared_columns, -start)

    # A policy whose objective is not fixed-width column overlap must not be judged by it.
    # These place the columns too big to ever be resident -- ClickBench's URL, Title and
    # Referer, which the trace shows serving 0.00 GiB however they are ordered -- and their
    # output scores worse on adjacency while being the thing under test. Judging them by the
    # old metric silently returned the caller's own order: big-same-adjacent from the natural
    # order came back unchanged, which read as "this policy is a no-op" when in fact its
    # result had been discarded.
    _objective_is_overlap = cfg.policy not in (
        "cacheable-first",
        "resident-overlap",
        "resident-overlap-spread",
        "big-same-adjacent",
        "big-diff-apart",
        "small-first-big-last",
        "small-first-big-runs",
        "small-first-big-ascending",
        "overlap-fit",
    )
    best_rank = (
        score(original, before, positions[0])
        if _objective_is_overlap
        else (float("-inf"), -1, -len(positions) - 1)
    )
    for start_position in start_positions:
        for path_builder in path_builders:
            path = path_builder(start_position, positions, signatures, cfg)
            candidate_queries = tuple(queries_by_position[pos] for pos in path)
            summary = sequence_overlap_summary(candidate_queries, cfg.scope)
            rank = score(candidate_queries, summary, start_position)
            if rank > best_rank:
                best_path = path
                best_queries = candidate_queries
                best_summary = summary
                best_rank = rank

    # Safeguard: if the reorder does not improve fixed-column adjacency, keep the given order rather
    # than risk making the workload worse.
    if best_queries == original:
        return finish(original, tuple(), before)
    if cfg.policy == "unfiltered-overlap":
        # This policy's objective is unfiltered-weighted overlap, not column count; revert only when
        # that is no better than the original order. Measured by column count, an order that
        # gathered more unfiltered columns would look like no improvement.
        if sequence_unfiltered_overlap(
            best_queries, cfg.scope
        ) <= sequence_unfiltered_overlap(original, cfg.scope):
            return finish(original, tuple(), before)
    elif cfg.policy not in (
        "byte-lru",
        "byte-overlap",
        "cacheable-first",
        "resident-overlap",
        "resident-overlap-spread",
        "big-same-adjacent",
        "big-diff-apart",
        "small-first-big-last",
        "small-first-big-runs",
        "small-first-big-ascending",
        "overlap-fit",
    ) and (
        best_summary.overlap_ratio <= before.overlap_ratio
        and best_summary.shared_columns <= before.shared_columns
    ):
        return finish(original, tuple(), before)

    resident_lru: list[ColumnKey] = []
    resident_set: set[ColumnKey] = set()
    previous_signature: frozenset[ColumnKey] = frozenset()
    pending_positions = list(best_path)
    decisions: list[ReorderDecision] = []

    for output_position, original_position in enumerate(best_path, start=1):
        qnum = queries_by_position[original_position]
        signature = signatures[original_position]
        score, resident_hits, previous_hits, future_overlap, footprint, hit_ratio = (
            _decision_score(
                original_position,
                signature,
                pending_positions,
                signatures,
                resident_set,
                previous_signature,
            )
        )
        decisions.append(
            ReorderDecision(
                position=output_position,
                selected_query=qnum,
                original_position=original_position,
                selected_score=float(score),
                resident_hit_columns=int(resident_hits),
                previous_overlap_columns=int(previous_hits),
                future_overlap_columns=int(future_overlap),
                query_columns=int(footprint),
                resident_hit_ratio=float(hit_ratio),
            )
        )
        pending_positions.remove(original_position)
        _touch_resident(
            resident_lru, resident_set, signature, cfg.resident_column_budget
        )
        previous_signature = signature

    return finish(best_queries, tuple(decisions), best_summary)


def write_decisions_csv(path: Path, workload_id: str, result: ReorderResult) -> None:
    """Write the reorder decisions to CSV: a pre-run explanation of why the order changed, not a
    performance log.
    """

    fields = [
        "workload_id",
        "policy",
        "scope",
        "reorder_elapsed_ms",
        "position",
        "selected_query",
        "original_position",
        "selected_score",
        "resident_hit_columns",
        "previous_overlap_columns",
        "future_overlap_columns",
        "query_columns",
        "resident_hit_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for decision in result.decisions:
            writer.writerow(
                {
                    "workload_id": workload_id,
                    "policy": result.config.policy,
                    "scope": result.config.scope,
                    "reorder_elapsed_ms": result.elapsed_ms,
                    "position": decision.position,
                    "selected_query": f"q{decision.selected_query}",
                    "original_position": decision.original_position,
                    "selected_score": decision.selected_score,
                    "resident_hit_columns": decision.resident_hit_columns,
                    "previous_overlap_columns": decision.previous_overlap_columns,
                    "future_overlap_columns": decision.future_overlap_columns,
                    "query_columns": decision.query_columns,
                    "resident_hit_ratio": decision.resident_hit_ratio,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    """Command-line options for running the reorder on its own, without Sirius."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries", required=True, help="Comma-separated query numbers, e.g. 8,1,16,3"
    )
    parser.add_argument("--policy", choices=POLICY_CHOICES, default="fixed-overlap")
    parser.add_argument("--scope", choices=SCOPE_CHOICES, default="fixed_width")
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="Candidates considered per step; 0 = the whole remaining batch",
    )
    parser.add_argument(
        "--keep-first", action="store_true", help="Keep the given first query first"
    )
    parser.add_argument(
        "--resident-column-budget",
        type=int,
        default=0,
        help="Diagnostic LRU resident column budget; 0 = unlimited",
    )
    parser.add_argument("--decisions-csv", type=Path, default=None)
    return parser


def main() -> int:
    """CLI entry point: print the order, before/after overlap and reorder time."""

    args = build_parser().parse_args()
    config = ReorderConfig(
        policy=args.policy,
        scope=args.scope,
        window=args.window,
        keep_first=args.keep_first,
        resident_column_budget=args.resident_column_budget,
    )
    result = reorder_query_sequence(parse_query_list(args.queries), config)
    print(f"original:  {format_query_sequence(result.original_queries)}")
    print(f"reordered: {format_query_sequence(result.reordered_queries)}")
    print(f"before overlap ratio: {result.before.overlap_ratio:.4f}")
    print(f"after  overlap ratio: {result.after.overlap_ratio:.4f}")
    print(f"reorder elapsed: {result.elapsed_ms:.3f} ms")
    if args.decisions_csv:
        write_decisions_csv(args.decisions_csv, "cli", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
