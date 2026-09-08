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
POLICY_CHOICES = ("none", "fixed-overlap", "byte-overlap", "byte-lru", "cost-ascending")

# {query: {table: [column, ...]}}, filled by load_query_table_columns().
# The union of a scan's projection AND filter columns, which is what the
# engine puts in an entry -- q6 projects 2 lineitem columns and caches 4.
QUERY_TABLE_COLUMNS: dict[str, dict[str, list[str]]] = {}

# {table: {column: bytes}}, filled by load_column_bytes(). Empty means "no size
# information", and every byte-weighted path falls back to counting columns.
COLUMN_BYTES: dict[str, dict[str, int]] = {}

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
    # 시작점은 컬럼이 가장 많은 쿼리 -- 이후 어떤 선택도 겹침을 만들기 쉬워지므로
    # 최악을 만들기에는 여기서 출발하는 편이 낫다.
    first = max(remaining, key=lambda q: (len(sig[q]), -q))
    order = [first]
    remaining.remove(first)
    while remaining:
        prev = sig[order[-1]]
        nxt = min(remaining, key=lambda q: (len(prev & sig[q]), q))
        order.append(nxt)
        remaining.remove(nxt)
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


def estimated_scan_bytes(qnum: int) -> int:
    """쿼리가 읽어야 하는 디코딩 후 바이트 추정치.

    SF100 실측 스캔 시간과 r=+0.64 (Spearman +0.63). 정확한 비용 모델은 아니지만
    plan 만으로 계산되므로, 측정된 실행 시간을 입력으로 요구하지 않는다.
    """

    per_table = QUERY_TABLE_COLUMNS.get(f"q{qnum}", {})
    return sum(column_bytes((table, column))
               for table, columns in per_table.items() for column in columns)


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

    if cfg.keep_first:
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
    if cfg.policy == "cost-ascending":
        # 시작점 탐색이 없다 -- 정렬 하나로 순서가 결정되므로, 아래의
        # "여러 start_position 중 최고를 고른다" 루프와 그 뒤의 overlap
        # 안전장치를 모두 건너뛴다.
        path = _cost_ascending_path(positions[0], positions, signatures, cfg)
        reordered = tuple(queries_by_position[pos] for pos in path)
        return finish(reordered, tuple(), sequence_overlap_summary(reordered, cfg.scope))
    if cfg.policy == "byte-lru":
        path_builder = _build_greedy_byte_lru_path
    elif cfg.policy == "byte-overlap":
        path_builder = _build_greedy_byte_overlap_path
    elif cfg.resident_column_budget > 0:
        path_builder = _build_greedy_resident_aware_path
    else:
        path_builder = _build_greedy_overlap_path

    def score(candidate_queries: tuple[int, ...], summary: SequenceOverlapSummary, start: int):
        if cfg.policy == "byte-overlap":
            shared = sum(
                sum(column_bytes(c) for c in query_signature(a, cfg.scope) & query_signature(b, cfg.scope))
                for a, b in zip(candidate_queries, candidate_queries[1:])
            )
            return (shared, summary.shared_columns, -start)
        if cfg.policy != "byte-lru":
            return (summary.overlap_ratio, summary.shared_columns, -start)
        served, _ = simulate_cache_hit_bytes(candidate_queries, cfg.scope, cfg.cache_budget_bytes)
        return (served, summary.shared_columns, -start)

    best_rank = score(original, before, positions[0])
    for start_position in start_positions:
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
    if cfg.policy not in ("byte-lru", "byte-overlap") and (
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
