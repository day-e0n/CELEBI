#!/usr/bin/env python3
"""온라인(실시간) resident-aware 쿼리 재정렬 프로토타입.

기존 cache_aware_query_reorder.py의 재정렬(run_random22_breakdown.py --reorder가
쓰는 것)은 22개 쿼리 중 하나도 실행하기 전에 전체 순서를 한 번에 확정하는
오프라인 방식이고, "지금 캐시에 뭐가 남아있는가"도 파이썬 안의 가상 LRU
시뮬레이션(cache_aware_query_reorder._touch_resident)일 뿐 실제 GPU 캐시 상태가
아니다.

이 스크립트는 같은 순위 함수(cache_aware_query_reorder._resident_aware_rank,
수식은 그대로 재사용)를 쓰되, 매 쿼리를 실제로 하나 실행한 직후 SiriusDB가 남긴
로그([fixed-page-cache] page_directory ..., auto_cache_skip ...)를 파싱해서
"지금 실제로 캐시에 남아있는 (table, column) 집합"을 재구성하고, 그걸로 다음
쿼리를 하나씩 골라 실행한다 -- 전체 순서를 미리 정하지 않고, 매 스텝마다 방금
끝난 실제 실행 결과를 보고 다음 걸 정하는 온라인 방식.

한계 (정확한 값이 아니라 근사치임을 밝힘):
SiriusDB에는 "지금 이 컬럼이 resident냐"를 바로 물어보는 SQL 인터페이스가 없어서,
이 스크립트는 로그를 리버스 엔지니어링해서 상태를 근사한다. 특히 global eviction
로그(page_budget applied scope=global, memory_pressure applied scope=global)는 어떤
엔트리의 페이지가 빠졌는지 엔트리 이름을 남기지 않는다 -- 그래서 이 스크립트는
"해당 엔트리를 마지막으로 건드렸을 때 관찰된 resident_pages"를 그대로 신뢰하고,
다른 엔트리 삽입이 유발한 글로벌 evict가 이 엔트리를 실제로 비웠는지는 반영하지
못한다. 그런 global eviction 로그 라인 개수는 세어서 마지막에 출력한다.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TPCH_DIR = REPO_ROOT / "test" / "tpch_performance"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(TPCH_DIR))

from performance_test import open_connection  # noqa: E402
from queries import QUERIES  # noqa: E402
from cache_aware_query_reorder import (  # noqa: E402
    ColumnKey,
    query_signature,
    reorder_query_sequence,
    ReorderConfig,
    _resident_aware_rank,
)
from run_random22_breakdown import ARRIVAL_ORDER, write_config, make_env  # noqa: E402

TIEBREAK_CHOICES = ("future", "previous", "smallest")


def rank_candidate(
    tiebreak: str,
    candidate_position: int,
    previous_position: int,
    remaining_positions: list[int],
    signatures: dict[int, frozenset[ColumnKey]],
    resident_set: frozenset[ColumnKey],
):
    """resident_ratio, resident_hits까지는 세 방식 모두 동일하고, 그 뒤 동점
    처리 기준만 다르다.

    future: 원래 수식 그대로(cache_aware_query_reorder._resident_aware_rank) --
      previous_overlap, 그다음 남은 모든 쿼리와의 future_overlap 순.
    previous: 남은 쿼리 전체를 훑는 future_overlap 계산 없이, 직전 쿼리와의
      overlap(O(1))까지만 본다.
    smallest: previous_overlap도 안 보고, 컬럼을 가장 적게 필요로 하는 쿼리를
      우선한다(v2 "가장 많이 필요한 쿼리 우선"의 반대 -- 캐시를 천천히 채워서
      eviction을 줄이자는 의도).
    """
    if tiebreak == "future":
        return _resident_aware_rank(
            candidate_position, previous_position, remaining_positions, signatures, resident_set
        )
    candidate = signatures[candidate_position]
    resident_hits = len(candidate & resident_set)
    resident_ratio = resident_hits / len(candidate) if candidate else 0.0
    if tiebreak == "previous":
        previous_overlap = len(signatures[previous_position] & candidate)
        return resident_ratio, resident_hits, previous_overlap, -candidate_position
    if tiebreak == "smallest":
        return resident_ratio, resident_hits, -len(candidate), -candidate_position
    raise ValueError(f"bad tiebreak: {tiebreak}")

TPCH_TABLES = {
    "customer", "lineitem", "nation", "orders",
    "part", "partsupp", "region", "supplier",
}

# entry name shape (auto_fixed_page_cache_name):
#   __wdy_auto_fixed_page:<path>;[<path>;...][filter=<sig>];cols=<c1>,<c2>,...,
ENTRY_TOUCH_RE = re.compile(
    r"\[fixed-page-cache\] page_directory \w+ table='(?P<name>.*)' fixed_cols=\d+ "
    r"pages=\d+ (?:pages_added=\d+ )?resident_pages=(?P<resident_pages>\d+)"
)
ADMISSION_SKIP_RE = re.compile(
    r"\[fixed-page-cache\] auto_cache_skip reason=(?P<reason>\S+) table='(?P<name>.*)'"
)
GLOBAL_EVICT_RE = re.compile(
    r"\[fixed-page-cache\] (page_budget applied scope=global|memory_pressure applied scope=global)"
)


def parse_entry_columns(entry_name: str) -> frozenset[ColumnKey]:
    """entry 이름(auto_fixed_page_cache_name 결과 문자열)에서 (table, column) 집합 복원."""
    cols_marker = ";cols="
    idx = entry_name.rfind(cols_marker)
    if idx == -1:
        return frozenset()
    cols_part = entry_name[idx + len(cols_marker):]
    columns = [c for c in cols_part.split(",") if c]

    prefix = "__wdy_auto_fixed_page:"
    body = entry_name[len(prefix):] if entry_name.startswith(prefix) else entry_name
    head = body[:idx - len(prefix)] if entry_name.startswith(prefix) else body[:idx]
    filter_idx = head.find(";filter=")
    paths_part = head[:filter_idx] if filter_idx != -1 else head
    paths = [p for p in paths_part.split(";") if p]

    table = None
    for path in paths:
        p = Path(path)
        for cand in (p.stem, p.parent.name):
            base = cand.split(".")[0]
            base_no_suffix = re.sub(r"_\d+$", "", base)
            if base_no_suffix in TPCH_TABLES:
                table = base_no_suffix
                break
        if table:
            break
    if table is None:
        return frozenset()
    return frozenset((table, col) for col in columns)


class ResidentModel:
    """로그를 리버스 엔지니어링해서 만드는 근사 resident 상태 (entry 단위)."""

    def __init__(self) -> None:
        self.entry_columns: dict[str, frozenset[ColumnKey]] = {}
        self.entry_resident: dict[str, bool] = {}
        self.unattributed_global_evictions = 0

    def feed_line(self, line: str) -> None:
        m = ENTRY_TOUCH_RE.search(line)
        if m:
            name = m.group("name")
            if name not in self.entry_columns:
                self.entry_columns[name] = parse_entry_columns(name)
            self.entry_resident[name] = int(m.group("resident_pages")) > 0
            return
        m = ADMISSION_SKIP_RE.search(line)
        if m:
            name = m.group("name")
            if name not in self.entry_columns:
                self.entry_columns[name] = parse_entry_columns(name)
            self.entry_resident[name] = False
            return
        if GLOBAL_EVICT_RE.search(line):
            self.unattributed_global_evictions += 1

    def resident_set(self) -> frozenset[ColumnKey]:
        cols: set[ColumnKey] = set()
        for name, resident in self.entry_resident.items():
            if resident:
                cols |= self.entry_columns.get(name, frozenset())
        return frozenset(cols)


class LogTailer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._pos = 0

    def new_lines(self) -> list[str]:
        with self.path.open("r", errors="replace") as f:
            f.seek(self._pos)
            lines = f.readlines()
            self._pos = f.tell()
        return lines


def run_query(con, qnum: int, execution: int) -> None:
    label = f"celebi_online_q{qnum}_exec{execution}"
    con.execute(f"CALL sirius_set_query_label('{label}')")
    con.execute("SET gpu_execution = true;")
    con.execute(QUERIES[f"q{qnum}"]).fetchall()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/mnt/nvme/sirius_tpch/tpch_parquet_sf50_optimized")
    parser.add_argument("--executions", type=int, default=3)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-budget", default="6GB")
    parser.add_argument("--min-free-bytes-per-gpu", default="4096MB")
    parser.add_argument("--gpu-usage-limit", default="20GB")
    parser.add_argument("--scope", default="fixed_width", choices=("fixed_width", "all_columns"))
    parser.add_argument("--tiebreak", default="future", choices=TIEBREAK_CHOICES)
    args = parser.parse_args()

    qnums = list(ARRIVAL_ORDER)
    signatures = {q: query_signature(q, args.scope) for q in qnums}

    offline_ref = reorder_query_sequence(
        qnums, ReorderConfig(policy="fixed-overlap", scope=args.scope, window=0,
                              keep_first=False, resident_column_budget=0)
    )
    print(f"[online] offline greedy reference order: "
          f"{','.join(f'q{q}' for q in offline_ref.reordered_queries)}", flush=True)

    output = args.output.resolve()
    case_dir = output / "celebi_online"
    log_dir = case_dir / "log_dir"
    config_path = case_dir / "sirius.yaml"
    write_config(config_path, case_dir / "telemetry_data", args.gpu_usage_limit)
    env = make_env("paging", args.devices, log_dir, config_path,
                    args.cache_budget, args.min_free_bytes_per_gpu)
    os.environ.update(env)

    con = open_connection(args.input, gpu_execution=True)
    tailer: LogTailer | None = None
    model = ResidentModel()
    log_parse_s = 0.0
    rank_decision_s = 0.0
    decision_count = 0
    try:
        for execution in range(1, args.executions + 1):
            remaining = list(qnums)
            start = remaining.pop(0)
            order_taken = [start]
            run_query(con, start, execution)
            print(f"[online] exec={execution} pos=1/{len(qnums)} q{start} (fixed start) done", flush=True)

            if tailer is None:
                candidates = sorted(log_dir.glob("sirius_*.log"))
                if not candidates:
                    raise RuntimeError(f"no sirius log file found under {log_dir}")
                tailer = LogTailer(candidates[0])

            t0 = time.perf_counter()
            for line in tailer.new_lines():
                model.feed_line(line)
            log_parse_s += time.perf_counter() - t0

            previous = start
            position = 2
            while remaining:
                t0 = time.perf_counter()
                resident = model.resident_set()
                best = max(
                    remaining,
                    key=lambda q: rank_candidate(
                        args.tiebreak, q, previous, remaining, signatures, resident
                    ),
                )
                rank_decision_s += time.perf_counter() - t0
                decision_count += 1
                hit = len(signatures[best] & resident)
                total = len(signatures[best])
                run_query(con, best, execution)
                print(f"[online] exec={execution} pos={position}/{len(qnums)} q{best} "
                      f"resident_hit={hit}/{total} done", flush=True)
                t0 = time.perf_counter()
                for line in tailer.new_lines():
                    model.feed_line(line)
                log_parse_s += time.perf_counter() - t0
                order_taken.append(best)
                remaining.remove(best)
                previous = best
                position += 1
            print(f"[online] exec={execution} order: "
                  f"{','.join(f'q{q}' for q in order_taken)}", flush=True)
    finally:
        con.close()

    print(f"[online] unattributed global-eviction log lines observed: "
          f"{model.unattributed_global_evictions}", flush=True)
    print(
        f"[online] reorder cost: {decision_count} decisions, "
        f"rank_decision_total={rank_decision_s * 1000:.2f}ms "
        f"(avg {rank_decision_s / decision_count * 1000:.4f}ms/decision), "
        f"log_parse_total={log_parse_s * 1000:.2f}ms "
        f"(avg {log_parse_s / decision_count * 1000:.4f}ms/decision) "
        f"-- log_parse is this prototype's log-reverse-engineering overhead, "
        f"not a cost a real cache-introspection API would pay",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
