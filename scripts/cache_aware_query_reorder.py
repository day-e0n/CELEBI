#!/usr/bin/env python3
"""고정 크기 컬럼 캐시를 의식해서 TPC-H 쿼리 순서를 재정렬하는 보조 스크립트.

이 스크립트는 읽기 전용 분석 쿼리 묶음을 실행하기 전에 순서만 바꾼다.
SQL 자체를 고치거나 쿼리 의미를 바꾸지는 않고, runner가 SiriusDB에 쿼리를
넘기기 전에 "고정 크기 컬럼을 더 많이 공유하는 쿼리끼리 붙여보자"는
실험용 스케줄링을 수행한다.

기본 기준은 fixed-page cache에 올라갈 수 있는 TPC-H fixed-width 컬럼이다.
즉 string처럼 variable-width 컬럼은 기본 overlap 계산에서 제외된다.
여기서 계산하는 hit/overlap은 실제 GPU page hit 로그가 아니라, 쿼리가
필요로 하는 컬럼 목록을 기반으로 한 실행 전 정적 지표다.
"""

from __future__ import annotations

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

# fixed-page cache 대상이 되는 fixed-width 컬럼 목록.
# pair latency 실험 스크립트와 같은 기준을 써야 그래프/워크로드 비교가 맞는다.
# string/varchar처럼 variable-width 컬럼은 여기서 빼고, 별도 segment/chunk 정책의
# 대상이라고 본다.
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
POLICY_CHOICES = ("none", "fixed-overlap", "byte-overlap", "byte-lru", "cost-ascending",
                  "cost-seeded-overlap", "unfiltered-overlap", "fixed-then-variable",
                  "fixed-first", "fixed-bytes-then-variable")

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

# 어떤 쿼리를 "무필터 쿼리"로 보고 앞쪽 구간에 넣을지. 컬럼의 이 비율 이상이
# 술어 없는 스캔에서 나오면 앞 구간이다. TPC-H SF50 에서 0.5 는 22개를 11 대 11
# 로 가른다 (q21/q11/q9 가 100%, q1/q6 이 0%).
UNFILTERED_PHASE_THRESHOLD = 0.5


_QUERY_AT: dict[int, int] = {}


@dataclass(frozen=True)
class ReorderConfig:
    """재정렬 정책 설정값.

    policy는 실제로 순서를 바꿀지 정하고, scope는 overlap 계산에 사용할 컬럼 범위를
    정한다. window는 greedy 탐색 후보 범위, resident_column_budget은 진단용 LRU
    column budget이다.
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
    """재정렬 결과에서 각 위치에 어떤 쿼리가 선택됐는지 남기는 진단 행.

    이 값들은 `query_reorder_report.csv`로 저장되어 "왜 이 순서가 됐는지"를
    설명할 때 쓰기 위한 자료다.
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
    """쿼리 시퀀스 전체의 인접 쿼리 간 컬럼 overlap 요약값."""

    shared_columns: int
    next_query_columns: int
    transition_count: int
    overlap_ratio: float


@dataclass(frozen=True)
class ReorderResult:
    """원래 쿼리 순서, 재정렬된 순서, before/after overlap을 묶은 결과."""

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
    """`1,2,6-8` 또는 `q1,q2,q6-q8` 같은 입력을 쿼리 번호 리스트로 바꾼다."""

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
    """쿼리 번호 리스트를 `q1,q2,...` 형태의 사람이 읽기 좋은 문자열로 바꾼다."""

    return ",".join(f"q{q}" for q in queries)


def load_column_set(module_name: str) -> str:
    """다른 벤치마크의 컬럼 맵으로 QUERY_COLUMNS / FIXED_WIDTH_COLUMNS를 갈아끼운다.

    이름을 rebind하지 않고 dict 내용을 in-place로 바꾼다. query_signature()가
    모듈 레벨 이름을 직접 읽기 때문에, rebind하면 이미 import된 쪽이 옛 객체를
    계속 보게 된다. performance_test.load_query_set()이 같은 이유로 같은 방식을
    쓴다.
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
    """쿼리 하나가 필요로 하는 `(table, column)` 집합을 만든다.

    `scope=fixed_width`이면 fixed-page cache 대상 컬럼만 남긴다. 그래서 쿼리에
    string 컬럼이 있어도 overlap 계산에서는 제외된다. `scope=all_columns`는
    쿼리 컬럼 overlap 자체를 보고 싶을 때 쓰는 비교용이다.
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


def worst_case_sequence(queries: Iterable[int], scope: str = "fixed_width") -> list[int]:
    """가장 불리한 도착 순서: 인접 쿼리끼리 컬럼이 최대한 안 겹치게 배치한다.

    재정렬이 무엇을 회복해 주는지 재려면 회복할 것이 있는 순서가 필요하다.
    벤치마크의 자연 순서는 우연히 이미 괜찮을 수 있고 (ClickBench 에서 재정렬
    효과가 -4.0%, TPC-H 에서 ±0.2% 로 작게 나온 이유이기도 하다), 그러면
    재정렬의 값이 과소평가된다. 이 순서는 그 반대쪽 끝 - "순서가 최악일 때"를
    준다.

    greedy: 남은 후보 중 직전 쿼리와 겹치는 컬럼이 가장 적은 것을 고른다.
    동점이면 쿼리 번호가 작은 쪽 (재현 가능하도록).
    """
    remaining = list(queries)
    if not remaining:
        return []
    sig = {q: query_signature(q, scope) for q in remaining}
    # 시작점은 컬럼이 가장 적은 쿼리.
    #
    # 이전 판은 컬럼이 가장 "많은" 쿼리에서 출발했다 -- 이후 선택에서 겹침을 낮게
    # 유지하기 쉽다는 이유였는데, 그건 overlap 지표만 본 판단이었다. 페이지 캐시에서
    # 엔트리는 컬럼을 누적하다 per-entry 상한에서 얼어붙고, 그 구성을 확정하는 것은
    # 엔트리를 처음 건드린 쿼리다. 따라서 가장 넓은 쿼리로 시작하는 것은 캐시에
    # 가장 유리하다: SF100 에서 그 순서(q8 선두, 16 컬럼)는 173.7s 로, 무작위 도착
    # 순서 평균 182.9s 보다 오히려 9s 빨랐다. docstring 의 의도와 반대였다.
    #
    # 가장 좁은 쿼리로 시작하면 엔트리가 좁게 열린 채 얼어붙는다 -- 실측된 최악의
    # 배치(q1 을 선두로 옮긴 것만으로 168.7s -> 195.5s)와 같은 메커니즘이다.
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


def sequence_overlap_summary(queries: Iterable[int], scope: str = "fixed_width") -> SequenceOverlapSummary:
    """시퀀스의 인접 쿼리들이 fixed-width 컬럼을 얼마나 공유하는지 계산한다.

    공식은 `sum(|prev ∩ next|) / sum(|next|)`이다. 이전 쿼리와 다음 쿼리 사이에
    겹치는 컬럼이 많을수록, 다음 쿼리가 이미 VRAM에 남아 있는 page를 재사용할
    가능성이 높다고 본다.
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


def worst_case_order(queries: Iterable[int], scope: str = "fixed_width") -> tuple[int, ...]:
    """의도적으로 overlap이 최소가 되는 순서를 만든다 (`_build_greedy_overlap_path`의 반대).

    각 단계에서 직전 쿼리와의 overlap ratio가 가장 낮은 후보를 고른다 -- CELEBI가
    "reorder 전" 상태로 잡아야 할, 인접 쿼리끼리 최대한 안 겹치는 최악의 baseline
    도착 순서를 만들 때 쓴다. 같은 템플릿(qnum) 반복도 최대한 서로 멀리 떨어뜨린다.
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
    """진단용 LRU resident column 집합을 갱신한다.

    실제 SiriusDB page directory를 조작하는 함수가 아니라, reorder report에
    "이 순서라면 앞에서 본 컬럼이 얼마나 남아 있을 법한가"를 적기 위한
    가벼운 시뮬레이션이다.
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
    """greedy 탐색에서 다음 후보 쿼리의 우선순위를 계산한다.

    우선순위는 이전 쿼리와의 overlap ratio, 공유 컬럼 수, 이후 남은 쿼리들과의
    미래 overlap, 원래 위치 tie-break 순서다.
    """

    previous = signatures[previous_position]
    candidate = signatures[candidate_position]
    shared = len(previous & candidate)
    ratio = shared / len(candidate) if candidate else 0.0
    future_overlap = sum(
        len(candidate & signatures[pos]) for pos in remaining_positions if pos != candidate_position
    )
    return ratio, shared, future_overlap, -candidate_position


def column_bytes(column: ColumnKey) -> int:
    """한 컬럼이 캐시에서 차지하는 바이트. 크기 정보가 없으면 1(= 개수 세기)."""

    table, name = column
    return COLUMN_BYTES.get(table, {}).get(name, 1)


def signature_bytes(signature: Iterable[ColumnKey]) -> int:
    return sum(column_bytes(column) for column in signature)


def load_column_bytes(path: str | Path) -> int:
    """probe_column_bytes.py가 만든 JSON을 COLUMN_BYTES에 적재한다.

    컬럼 "개수" 겹침은 25행짜리 nation.n_nationkey를 6억행 lineitem.l_orderkey와
    같은 1로 세기 때문에, byte-lru 정책은 실측 크기가 있어야 의미가 있다.
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
    """바이트 예산 LRU를 돌려 (캐시로 서빙된 바이트, 전체 요구 바이트)를 낸다.

    인접 쌍 overlap과 달리 (1) 첫 쿼리는 빈 캐시를 만나 히트가 0으로 계산되고,
    (2) 컬럼이 여러 쿼리를 건너 살아남는 것을 반영하며, (3) 컬럼 크기로 가중된다.
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
    """바이트 예산 LRU를 시뮬레이션하며 "캐시에서 가장 많은 바이트를 받아갈"
    쿼리를 다음으로 고른다.

    `_build_greedy_overlap_path`는 직전 쿼리와 겹치는 컬럼 개수만 보므로
    lineitem 컬럼 하나(4.5 GB)와 nation 컬럼 하나(100 B)를 구분하지 못하고,
    이미 밀려난 컬럼도 여전히 겹친 것으로 센다."""

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
    """probe_scan_requests.py가 만든 JSON을 적재한다 (cost-ascending 전용)."""

    import json

    QUERY_TABLE_COLUMNS.clear()
    QUERY_TABLE_COLUMNS.update(json.loads(Path(path).read_text()))
    return len(QUERY_TABLE_COLUMNS)


def load_query_table_filters(path: str | Path) -> int:
    """extract_query_filters.py가 만든 JSON을 적재한다 (unfiltered-overlap 전용)."""

    import json

    QUERY_TABLE_FILTERS.clear()
    QUERY_TABLE_FILTERS.update(json.loads(Path(path).read_text()))
    return len(QUERY_TABLE_FILTERS)


def scan_is_unfiltered(qnum: int, table: str) -> bool:
    """쿼리 `qnum`이 테이블 `table`을 술어 없이 읽는가.

    필터 정보가 없으면 모두 무필터로 본다 -- 그래야 JSON을 주지 않았을 때 기존
    컬럼 전용 동작과 정확히 같아진다.
    """

    if not QUERY_TABLE_FILTERS:
        return True
    per_table = QUERY_TABLE_FILTERS.get(f"q{qnum}")
    if per_table is None or table not in per_table:
        return True
    return not per_table[table]


def unfiltered_signature(qnum: int, scope: str = "fixed_width") -> frozenset[ColumnKey]:
    """`query_signature` 중 무필터 스캔에서 나오는 컬럼만."""

    return frozenset((table, column) for table, column in query_signature(qnum, scope)
                     if scan_is_unfiltered(qnum, table))


def estimated_scan_bytes(qnum: int) -> int:
    """쿼리가 읽어야 하는 디코딩 후 바이트 추정치.

    SF100 실측 스캔 시간과 r=+0.64 (Spearman +0.63). 정확한 비용 모델은 아니지만
    plan 만으로 계산되므로, 측정된 실행 시간을 입력으로 요구하지 않는다.
    """

    per_table = QUERY_TABLE_COLUMNS.get(f"q{qnum}", {})
    return sum(column_bytes((table, column))
               for table, columns in per_table.items() for column in columns)


def _cost_seeded_overlap_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """가장 싼 쿼리에서 출발해, 그 뒤로는 overlap greedy 로 잇는다.

    `fixed-overlap` 은 시작점을 "전체 overlap ratio 가 가장 커지는 자리"로 고르는데,
    그 목적함수의 분모(`sum(|next|)`)에는 첫 쿼리가 들어가지 않는다. 첫 자리는
    공짜로 보이고, 그래서 SF100 에서 가장 비싼 q21(baseline scan 22.2s)이 1번으로
    간다 -- 캐시가 비어 히트가 보장되지 않는 유일한 자리인데.

    여기서는 그 한 자리만 비용으로 고정한다. 콜드 구간을 가장 싼 스캔으로 소모하고,
    나머지 21개는 그대로 overlap 이 정한다.
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
    """스캔 비용 오름차순. 인접 overlap 을 아예 보지 않는다.

    이 캐시에서 순서가 중요한 이유는 인접성이 아니라 누적이다: 엔트리는 컬럼을
    쌓아가다 per-entry 상한(budget/2)에서 얼어붙고, 그 뒤로는 아무도 넓히지
    못한다 (`admission_stop_widening` 이 런당 480-732회). 따라서 비싼 스캔은
    엔트리가 최대한 넓어진 뒤에 도착해야 하고, 싼 스캔이 먼저 와서 그것을
    넓혀 주어야 한다.

    SF100 에서 fixed-overlap 186.0s, 무작위 도착 평균 182.9s, 이 정책 168.7s.
    """

    del start_position, signatures, cfg
    return sorted(positions, key=lambda pos: (estimated_scan_bytes(_QUERY_AT[pos]), pos))


def _byte_pair_overlap_rank(
    previous_position: int,
    candidate_position: int,
    remaining_positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
) -> tuple[float, int, int, int]:
    """`_pair_overlap_rank`와 같되 컬럼을 개수가 아니라 바이트로 잰다.

    캐시 시뮬레이션을 하지 않는다는 점이 `byte-lru`와의 차이다. 잔존 집합을
    모델링하면 그 모델이 틀렸을 때 오차가 그대로 순서에 들어가므로, 여기서는
    "직전 쿼리와 겹치는 바이트"라는 관측 가능한 양만 쓴다.
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
    """직전 쿼리와 바이트 겹침이 가장 큰 쿼리를 하나씩 잇는다."""

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        candidate_pool = remaining[:window]
        previous_position = path[-1]
        best_position = max(
            candidate_pool,
            key=lambda pos: _byte_pair_overlap_rank(previous_position, pos, remaining, signatures),
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
    """fixed-width 겹침을 1순위로, variable-width 겹침을 2순위로 둔다.

    `fixed-overlap`은 scope 가 fixed_width 라 string 컬럼을 아예 안 본다. 그래서
    ClickBench 처럼 URL 하나가 나머지 컬럼을 합친 것보다 큰 workload 에서는 두
    string 쿼리를 붙일지 말지가 순서에 전혀 반영되지 않는다. `byte-overlap`은
    반대로 string 이 모든 결정을 지배한다. 이 정책은 fixed 가 같은 점수일 때만
    string 이 순서를 정하게 해서, fixed 재사용을 먼저 확보하고 남은 자유도로
    string 재사용을 챙긴다.
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
    """그 자리 쿼리가 읽는 variable-width(문자열) 컬럼 집합."""

    qnum = _QUERY_AT[position] if _QUERY_AT else position
    return query_signature(qnum, "all_columns") - query_signature(qnum, "fixed_width")


def _build_fixed_then_variable_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """fixed 겹침이 같으면 string 겹침이 큰 쪽을 먼저 잇는다."""

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        candidate_pool = remaining[:window]
        previous_position = path[-1]
        best_position = max(
            candidate_pool,
            key=lambda pos: _fixed_then_variable_pair_overlap_rank(
                previous_position, pos, remaining, signatures),
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
    """무필터 스캔이 남긴 컬럼을 우선해서 다음 쿼리를 고르는 순위.

    `_pair_overlap_rank`는 공유 컬럼을 전부 1로 세는데, 캐시 입장에서 그 둘은
    같지 않다. 앞 쿼리가 술어 없이 읽은 컬럼은 뒤에 오는 누구든 쓸 수 있고,
    술어를 걸고 읽은 컬럼은 그보다 좁은 술어를 가진 쿼리만 쓸 수 있다. 그래서
    공유 컬럼의 가중치를 "그 컬럼을 캐시에 남기는 쪽"(= previous)의 스캔이
    무필터였는지로 나눈다.
    """

    previous_query = _QUERY_AT[previous_position]
    candidate = signatures[candidate_position]
    shared = signatures[previous_position] & candidate

    def weight(column: ColumnKey) -> float:
        # 캐시에 남기는 쪽은 previous 다. 그 스캔이 무필터면 온전한 1점.
        return 1.0 if scan_is_unfiltered(previous_query, column[0]) else FILTERED_OVERLAP_WEIGHT

    weighted = sum(weight(column) for column in shared)
    ratio = weighted / len(candidate) if candidate else 0.0
    # 동점 해소: 이 후보가 "앞으로" 남길 무필터 컬럼이 많을수록 좋다.
    candidate_query = _QUERY_AT[candidate_position]
    future = sum(
        len(unfiltered_signature(candidate_query) & signatures[pos])
        for pos in remaining_positions if pos != candidate_position
    )
    return ratio, weighted, float(future), -candidate_position


def unfiltered_policy_applies(queries: Iterable[int]) -> bool:
    """무필터 우선 재정렬이 이 워크로드에서 구분할 것이 있는가.

    이 정책은 "술어 없이 읽은 페이지는 뒤에 오는 누구나 쓸 수 있다"를 근거로
    순서를 정한다. 그런데 스캔 크기 게이트가 열리면 조인의 동적 필터를 단
    스캔도 필터 이전 데이터를 캐싱한다 -- 그런 페이지에는 술어가 안 붙는다.
    그러므로 지배 테이블을 **조인 없이** 읽는 쿼리가 하나도 없으면 그 테이블의
    페이지는 전부 술어 없이 캐시에 들어가고, 정책이 가르는 기준이 그 테이블에
    대해 아무 정보도 담지 않는다. 그때는 순서만 흔들어 손해가 난다.

    측정된 값 (지배 테이블을 단독으로 읽는 쿼리의 비율):
      ClickBench 41/41 = 100%   재정렬 -9.1%p 이득
      TPC-H SF50  2/17 = 11.8%  재정렬 -8.5%p 이득  (q1, q6 이 단독으로 읽는다)
      SSB SF50    0/13 =  0.0%  재정렬 +6.5%p 손해

    그래서 문턱은 "하나라도 있는가"다. 비율이 아니라 존재 여부인 것은, 캐시를
    술어 없는 페이지로 채워 줄 쿼리는 한 개만 있어도 되기 때문이다 -- TPC-H 가
    11.8% 로 이득을 내는 이유가 그것이다. 필터 정보가 없으면 판단할 수 없으므로
    참을 돌려 기존 동작을 유지한다.
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
    """쿼리 컬럼 중 술어 없는 스캔에서 나오는 비율."""

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
    """`start_position`에서 출발해 pool 을 무필터 가중 greedy 로 훑는다."""

    path = [start_position]
    remaining = [pos for pos in pool if pos != start_position]
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        previous_position = path[-1]
        best_position = max(
            remaining[:window],
            key=lambda pos: _unfiltered_pair_overlap_rank(
                previous_position, pos, remaining, signatures),
        )
        path.append(best_position)
        remaining.remove(best_position)
    return path


def _fixed_share(position: int) -> float:
    """그 자리 쿼리가 읽는 바이트 중 fixed-width 컬럼이 차지하는 비율."""

    qnum = _QUERY_AT[position] if _QUERY_AT else position
    fixed = query_signature(qnum, "fixed_width")
    every = query_signature(qnum, "all_columns")
    total = sum(column_bytes(c) for c in every)
    if total <= 0:
        # 크기 정보가 없으면 컬럼 개수로 돌아간다.
        return len(fixed) / len(every) if every else 0.0
    return sum(column_bytes(c) for c in fixed) / total


FIXED_PHASE_THRESHOLD = 0.5


def _build_fixed_first_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """fixed-width 컬럼이 주인 쿼리를 앞 구간에 몰고, string 이 주인 쿼리를 뒤에 붙인다.

    `fixed-overlap` 계열은 매 단계 순위만 fixed 기준으로 매기므로 string 쿼리가
    중간중간 끼어든다. ClickBench 에서 이게 문제인 이유는 측정으로 나왔다: 재정렬을
    하면 fixed 컬럼이 캐시에 들어가는 양 자체가 15.4GiB 에서 8.6GiB 로 줄고, 서빙도
    7.52GiB 에서 6.19GiB 로 준다. 축출이 fixed 를 건드려서가 아니라 -- variable 우선
    축출에서 fixed 는 0 바이트 버려졌다 -- string 쿼리가 먼저 자리를 채워 fixed 가
    들어갈 공간이 없기 때문이다. 그래서 순위가 아니라 구간을 나눈다: 캐시가 빈 동안
    fixed 쿼리들이 먼저 자리를 잡고 서로 재사용한 뒤, string 쿼리를 태운다.
    """

    front = [pos for pos in positions if _fixed_share(pos) >= FIXED_PHASE_THRESHOLD]
    back = [pos for pos in positions if pos not in set(front)]
    if not front or not back:
        return _build_greedy_overlap_path(start_position, positions, signatures, cfg)
    if start_position not in front:
        front = [start_position] + front
        back = [pos for pos in back if pos != start_position]
    path = _greedy_within(start_position, front, signatures, cfg)
    bridge = max(back, key=lambda pos: _pair_overlap_rank(path[-1], pos, back, signatures))
    path.extend(_greedy_within(bridge, back, signatures, cfg))
    return path


def _fixed_bytes_then_variable_rank(
    previous_position: int,
    candidate_position: int,
    remaining_positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
) -> tuple[int, float, float, float, int]:
    """fixed 겹침 바이트가 1순위, 동점이면 variable 겹침이 가장 적은 쪽, fixed 가 0이면 variable.

    `fixed-overlap` 은 겹침을 컬럼 개수로 세므로 ClickBench 처럼 한 컬럼이 나머지를
    합친 것보다 큰 workload 에서 빗나가고, `byte-overlap` 은 반대로 string 이 모든
    결정을 지배한다. 여기서는 실제로 재사용되는 쪽 -- 열 row group 이 예산에 다 들어가고
    여러 쿼리가 읽는 컬럼, 측정상 거의 전부 fixed-width -- 의 겹침 바이트를 먼저 쌓고,
    그것이 같을 때는 string 겹침이 **적은** 후보를 고른다. string 을 붙여봐야 그 페이지는
    재사용 전에 밀려나고 (URL 은 43.8GiB 를 넣어 0.41GiB 만 서빙했다) 그 사이 fixed 를
    쓸어내기 때문이다. fixed 겹침이 바닥난 뒤에야 string 겹침으로 정렬한다.
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
        # 1순위 fixed 바이트, 2순위 string 겹침이 적은 쪽 (부호를 뒤집어 최대화에 태운다).
        return 1, fixed_shared, -var_shared, future, -candidate_position
    # fixed 로 이을 것이 없으면 string 겹침으로 잇는다.
    return 0, 0.0, var_shared, future, -candidate_position


def _build_fixed_bytes_then_variable_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """`_fixed_bytes_then_variable_rank` 로 한 쿼리씩 잇는다."""

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        candidate_pool = remaining[:window]
        previous_position = path[-1]
        best_position = max(
            candidate_pool,
            key=lambda pos: _fixed_bytes_then_variable_rank(
                previous_position, pos, remaining, signatures),
        )
        path.append(best_position)
        remaining.remove(best_position)
    return path


def _build_unfiltered_first_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """술어 없이 읽는 쿼리를 앞 구간에 몰고, 술어를 건 쿼리를 뒤에 붙인다.

    컬럼 겹침만 보는 greedy 는 `l_shipdate < 1995` 와 `l_shipdate >= 1998` 을 완전히
    겹치는 쌍으로 세는데, 둘은 서로의 페이지를 한 행도 못 쓴다. 술어 없이 읽은
    페이지만이 뒤에 오는 누구든 읽을 수 있으므로, 그런 쿼리를 먼저 돌려 캐시를
    "아무나 쓸 수 있는" 내용물로 채운 뒤 좁은 술어를 가진 쿼리를 태운다.
    """

    front = [pos for pos in positions
             if unfiltered_fraction(_QUERY_AT[pos], cfg.scope) >= UNFILTERED_PHASE_THRESHOLD]
    back = [pos for pos in positions if pos not in set(front)]
    if start_position not in front:
        # 출발점은 재정렬 쪽에서 무필터 쿼리로 제한된다. 그래도 들어오면 그 쿼리를
        # 앞 구간에 넣어서 "무필터 먼저"라는 정책 자체는 유지한다.
        front = [start_position] + front
        back = [pos for pos in back if pos != start_position]
    path = _greedy_within(start_position, front, signatures, cfg)
    if back:
        bridge = max(back, key=lambda pos: _unfiltered_pair_overlap_rank(
            path[-1], pos, back, signatures))
        path.extend(_greedy_within(bridge, back, signatures, cfg))
    return path


def sequence_unfiltered_overlap(queries: Iterable[int], scope: str = "fixed_width") -> float:
    """인접 쌍의 가중 공유 컬럼 합. unfiltered-overlap 정책의 목적함수."""

    sequence = list(queries)
    total = 0.0
    for previous, nxt in zip(sequence, sequence[1:]):
        shared = query_signature(previous, scope) & query_signature(nxt, scope)
        total += sum(1.0 if scan_is_unfiltered(previous, table) else FILTERED_OVERLAP_WEIGHT
                     for table, _ in shared)
    return total


def _build_greedy_overlap_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """특정 시작 위치에서 출발해 overlap이 큰 다음 쿼리를 하나씩 고르는 경로를 만든다."""

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        candidate_pool = remaining[:window]
        previous_position = path[-1]
        best_position = max(
            candidate_pool,
            key=lambda pos: _pair_overlap_rank(previous_position, pos, remaining, signatures),
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
    """budget-aware greedy 탐색에서 다음 후보 쿼리의 우선순위를 계산한다.

    직전 쿼리와의 overlap 대신, "지금 캐시에 실제로 남아있는 컬럼"과 얼마나
    겹치는지를 최우선으로 본다 — budget이 좁아서 이미 밀려난 컬럼은 아무리
    직전 쿼리와 겹쳐도 소용없기 때문이다.
    """

    candidate = signatures[candidate_position]
    previous = signatures[previous_position]
    resident_hits = len(candidate & resident_set)
    resident_ratio = resident_hits / len(candidate) if candidate else 0.0
    previous_overlap = len(previous & candidate)
    future_overlap = sum(
        len(candidate & signatures[pos]) for pos in remaining_positions if pos != candidate_position
    )
    return resident_ratio, resident_hits, previous_overlap, future_overlap, -candidate_position


def _build_greedy_resident_aware_path(
    start_position: int,
    positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    cfg: ReorderConfig,
) -> list[int]:
    """resident cache(용량 제한 LRU)를 시뮬레이션하면서 다음 쿼리를 고르는 경로를 만든다.

    `_build_greedy_overlap_path`는 직전 쿼리와의 overlap만 보고, 실제로 그
    컬럼이 캐시에 아직 남아있는지는 신경 쓰지 않는다. 이 버전은 매 단계마다
    `_touch_resident`로 resident 상태를 갱신하면서, "지금 진짜 캐시에 있는 것과
    가장 많이 겹치는" 쿼리를 우선 선택한다.
    """

    path = [start_position]
    remaining = [pos for pos in positions if pos != start_position]
    resident_lru: list[ColumnKey] = []
    resident_set: set[ColumnKey] = set()
    _touch_resident(resident_lru, resident_set, signatures[start_position], cfg.resident_column_budget)

    while remaining:
        window = len(remaining) if cfg.window <= 0 else min(cfg.window, len(remaining))
        candidate_pool = remaining[:window]
        previous_position = path[-1]
        best_position = max(
            candidate_pool,
            key=lambda pos: _resident_aware_rank(pos, previous_position, remaining, signatures, resident_set),
        )
        path.append(best_position)
        remaining.remove(best_position)
        _touch_resident(resident_lru, resident_set, signatures[best_position], cfg.resident_column_budget)
    return path


def _decision_score(
    original_position: int,
    signature: frozenset[ColumnKey],
    pending_positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    resident_set: set[ColumnKey],
    previous_signature: frozenset[ColumnKey],
) -> tuple[float, int, int, int, int, float]:
    """CSV 리포트용 점수와 hit 진단값을 계산한다.

    실제 reorder의 핵심 기준은 `_pair_overlap_rank`와 전체 overlap summary다.
    이 score는 사람이 결과를 읽을 때 resident hit, 직전 쿼리 overlap, future
    overlap을 한 행에서 같이 보기 위한 설명용 수치다.
    """

    resident_hits = len(signature & resident_set)
    previous_hits = len(signature & previous_signature)
    future_overlap = sum(
        len(signature & signatures[pos]) for pos in pending_positions if pos != original_position
    )
    footprint = len(signature)
    misses = max(footprint - resident_hits, 0)
    score = 4.0 * resident_hits + 2.0 * previous_hits + 0.25 * future_overlap - 0.10 * misses
    hit_ratio = resident_hits / footprint if footprint else 0.0
    return score, resident_hits, previous_hits, future_overlap, footprint, hit_ratio


def reorder_query_sequence(queries: Iterable[int], config: ReorderConfig | None = None) -> ReorderResult:
    """쿼리 시퀀스를 cache-aware 순서로 재정렬한다.

    가능한 시작점을 바꿔가며 greedy 경로를 만들고, 전체 fixed-column overlap이
    가장 큰 순서를 선택한다. 다만 overlap이 좋아지지 않으면 안전장치로 원래
    사용자 입력 순서를 유지한다.
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
        return ReorderResult(original, reordered, cfg, decisions, before, after, elapsed_ms)

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
        # 이 워크로드에서는 술어의 유무가 재사용성을 가르지 못한다. 순서를 그대로 둔다.
        return finish(original, tuple(), before)
    if cfg.policy == "unfiltered-overlap" and not cfg.keep_first:
        # 첫 쿼리가 캐시의 첫 내용물을 정한다. 술어를 건 스캔으로 열면 뒤따르는
        # 쿼리 대부분이 그 페이지를 못 읽으니, 무필터 스캔을 가진 쿼리로만
        # 출발점을 고르고 그 중 무필터 컬럼이 넓은 것부터 시도한다.
        seeds = [pos for pos in positions
                 if unfiltered_fraction(queries_by_position[pos], cfg.scope)
                 >= UNFILTERED_PHASE_THRESHOLD]
        start_positions = sorted(
            seeds or positions,
            key=lambda pos: (-unfiltered_fraction(queries_by_position[pos], cfg.scope),
                             -len(query_signature(queries_by_position[pos], cfg.scope)), pos))
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
        return finish(reordered, tuple(), sequence_overlap_summary(reordered, cfg.scope))
    if cfg.policy == "cost-ascending":
        # 시작점 탐색이 없다 -- 정렬 하나로 순서가 결정되므로, 아래의
        # "여러 start_position 중 최고를 고른다" 루프와 그 뒤의 overlap
        # 안전장치를 모두 건너뛴다.
        path = _cost_ascending_path(positions[0], positions, signatures, cfg)
        reordered = tuple(queries_by_position[pos] for pos in path)
        return finish(reordered, tuple(), sequence_overlap_summary(reordered, cfg.scope))
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
    elif cfg.resident_column_budget > 0:
        path_builders = [_build_greedy_resident_aware_path]
    else:
        path_builders = [_build_greedy_overlap_path]

    def score(candidate_queries: tuple[int, ...], summary: SequenceOverlapSummary, start: int):
        if cfg.policy == "byte-overlap":
            shared = sum(
                sum(column_bytes(c) for c in query_signature(a, cfg.scope) & query_signature(b, cfg.scope))
                for a, b in zip(candidate_queries, candidate_queries[1:])
            )
            return (shared, summary.shared_columns, -start)
        if cfg.policy == "unfiltered-overlap":
            return (sequence_unfiltered_overlap(candidate_queries, cfg.scope),
                    summary.shared_columns, -start)
        if cfg.policy != "byte-lru":
            return (summary.overlap_ratio, summary.shared_columns, -start)
        served, _ = simulate_cache_hit_bytes(candidate_queries, cfg.scope, cfg.cache_budget_bytes)
        return (served, summary.shared_columns, -start)

    best_rank = score(original, before, positions[0])
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

    # 안전장치: 자동 재정렬이 fixed-column adjacency locality를 개선하지 못하면
    # 괜히 workload를 악화시키지 않도록 사용자가 준 원래 순서를 그대로 쓴다.
    if best_queries == original:
        return finish(original, tuple(), before)
    if cfg.policy == "unfiltered-overlap":
        # 이 정책의 목적함수는 컬럼 개수가 아니라 무필터 가중 겹침이다. 원래
        # 순서보다 그 값이 크지 않을 때만 되돌린다 -- 컬럼 개수 기준으로 재면
        # 무필터 컬럼을 더 모은 순서를 개선 없음으로 오판해 버린다.
        if sequence_unfiltered_overlap(best_queries, cfg.scope) <= sequence_unfiltered_overlap(
                original, cfg.scope):
            return finish(original, tuple(), before)
    elif cfg.policy not in ("byte-lru", "byte-overlap") and (
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
        score, resident_hits, previous_hits, future_overlap, footprint, hit_ratio = _decision_score(
            original_position,
            signature,
            pending_positions,
            signatures,
            resident_set,
            previous_signature,
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
        _touch_resident(resident_lru, resident_set, signature, cfg.resident_column_budget)
        previous_signature = signature

    return finish(best_queries, tuple(decisions), best_summary)


def write_decisions_csv(path: Path, workload_id: str, result: ReorderResult) -> None:
    """재정렬 의사결정 과정을 CSV로 저장한다.

    이 파일은 실행 성능 로그가 아니라, "어떤 기준으로 쿼리 순서를 바꿨는지"를
    보여주는 실행 전 설명 자료다.
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
    """단독 실행용 CLI 옵션을 정의한다.

    SiriusDB를 실제로 실행하지 않고 reorder 결과만 빠르게 확인할 때 사용한다.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, help="쉼표로 구분한 쿼리 번호. 예: 8,1,16,3")
    parser.add_argument("--policy", choices=POLICY_CHOICES, default="fixed-overlap")
    parser.add_argument("--scope", choices=SCOPE_CHOICES, default="fixed_width")
    parser.add_argument("--window", type=int, default=0, help="매 단계에서 볼 후보 범위. 0이면 전체 남은 batch를 본다")
    parser.add_argument("--keep-first", action="store_true", help="사용자가 보낸 첫 쿼리를 첫 실행으로 고정한다")
    parser.add_argument("--resident-column-budget", type=int, default=0, help="진단용 LRU resident column budget. 0이면 제한 없음")
    parser.add_argument("--decisions-csv", type=Path, default=None)
    return parser


def main() -> int:
    """CLI 시작점. 쿼리 순서, before/after overlap, reorder 시간을 출력한다."""

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
