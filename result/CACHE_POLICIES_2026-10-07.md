# 현재 query reorder와 page eviction 정책 — 2026-10-07

현재 A/HIT 실험의 reorder는 **배치 안의 SQL 실행 순서**를 바꾼다. SQL이나 JOIN/aggregation/sort의 연산자 순서를 바꾸는 기능은 아니다. 실행 전에 정적인 쿼리별 `(table, column)` 목록을 이용하며 GPU cache의 실제 resident 상태를 읽지 않는다. [범위](/home/dy1013/sirius_multi-gpu/scripts/cache_aware_query_reorder.py:4), [컬럼 목록](/home/dy1013/sirius_multi-gpu/scripts/cache_aware_query_reorder.py:210).

## 실제 실행한 설정

| 조건 | SQL 순서 | eviction hint |
|---|---|---|
| A baseline / HIT base | 원래 arrival | 없음, GPU page reuse/populate OFF |
| A lru / HIT lru | 원래 arrival | OFF: variable-width 우선, 그룹 안에서 LRU |
| HIT ro_lru | 재정렬 | 위와 같은 eviction |
| A best / HIT ro_evict | 재정렬 | `evict`: 다음 사용이 먼 컬럼의 page부터 |
| HIT evict | 원래 arrival | 같은 `evict` |

재정렬 조건은 모두 `small-first-big-last`, column metadata는 `experiment/frag_replay/tpch_sf50_column_bytes.json` 또는 `clickbench_column_bytes.json`이다. runner 기본 `fixed-overlap`이 아니라 launcher에서 이 정책을 명시했다. GPU 1, TH SF50 4GiB, CB 6GiB, free pool floor 6GiB, execution 3회다. 기존 CB는 q28/q29를 제외한 **41개**이고 q24는 포함한다. A best/HIT ro_evict의 future horizon은 기본 `execution`이다. `A_*_bestrun`의 `run` horizon은 별도 조건이다. [A launcher](/tmp/claude-1002/-home-dy1013-sirius-multi-gpu/9134a118-eb26-4f12-844a-fccab0a76115/scratchpad/run_A.sh:1), [공통 설정](/tmp/claude-1002/-home-dy1013-sirius-multi-gpu/9134a118-eb26-4f12-844a-fccab0a76115/scratchpad/runlib.sh:7), [HIT launcher](/tmp/claude-1002/-home-dy1013-sirius-multi-gpu/9134a118-eb26-4f12-844a-fccab0a76115/scratchpad/run_hit.sh:9), [runner 설정](/home/dy1013/sirius_multi-gpu/scripts/run_clickbench_breakdown.py:197).

저장 launcher에는 `SIRIUS_PAGE_EVICT_ORDER=size`, big-prefix, oversized-column skip, replace-admission override가 없다. 전체 상속 환경을 저장한 것은 아니므로 과거의 모든 ambient env가 없었다고 증명할 수는 없다. 기본 ratio 1.0과 저장 metadata로 순수 Python 재계산한 TH/CB 순서는 실제 runner 로그와 모두 정확히 일치했다. 이 확인은 DB나 벤치마크를 실행하지 않았다.

## Reorder가 고르는 순서

“큰 컬럼”은 **컬럼 전체 decoded bytes > cache budget × ratio**이며 ratio 기본값은 1.0이다. 이는 컬럼 전체가 동시에 상주할 수 없다는 분류다. 그 컬럼의 일부 page도 cache hit할 수 없다는 뜻은 아니다. 이 metadata/예산에서 사용 중인 큰 컬럼은 TH의 `l_shipinstruct` 약4.470GiB, CB의 `URL` 약9.272GiB, `Title` 약9.161GiB, `Referer` 약6.790GiB다. 문자열 크기는 metadata의 추정값이며 최신 실측 allocation/physical I/O가 아니다. [분류](/home/dy1013/sirius_multi-gpu/scripts/cache_aware_query_reorder.py:920).

1. 큰 컬럼을 읽지 않는 쿼리를 앞에 둔다. 직전 쿼리와 공유하는 작은 컬럼 bytes / 후보의 작은 컬럼 bytes를 먼저 비교하고, 동점이면 공유 bytes, 남은 쿼리들과 공유할 bytes의 합, 원래 arrival 위치 순으로 고른다.
2. 큰 컬럼을 읽는 뒤 구간은 직전과 같은 큰 컬럼 집합, 공유하는 큰 컬럼 bytes, 원래 arrival 위치 순으로 고른다.
3. `window=0`, `keep_first=False`이므로 여러 시작점에서 greedy 순서를 만든다. **최종 시작점 선택은 여전히 fixed-width 컬럼 개수 기준**의 전체 인접 overlap ratio → 공유 컬럼 개수 → 원래 시작 위치다. 실제 cache hit를 최적화한 해나 전역 최적해는 아니다.

[작은 컬럼 점수](/home/dy1013/sirius_multi-gpu/scripts/cache_aware_query_reorder.py:984), [앞/뒤 구간 규칙](/home/dy1013/sirius_multi-gpu/scripts/cache_aware_query_reorder.py:1220), [뒤 구간 동점](/home/dy1013/sirius_multi-gpu/scripts/cache_aware_query_reorder.py:1245), [시작점/최종 점수](/home/dy1013/sirius_multi-gpu/scripts/cache_aware_query_reorder.py:1550), [최종 선택](/home/dy1013/sirius_multi-gpu/scripts/cache_aware_query_reorder.py:1607).

이 정책은 인접 overlap 개선이 없을 때 원래 순서로 되돌리는 guard의 예외다. 기존 TH fixed-column overlap은7.19→62.09%, CB는51.90→48.00%다. 이 수치는 정적 컬럼 개수 지표이며 측정한 GPU cache hit rate가 아니다. 큰 컬럼을 뒤로 보내는 규칙과 최종 seed 점수가 서로 다른 기준인 점에 주의한다. [guard 예외](/home/dy1013/sirius_multi-gpu/scripts/cache_aware_query_reorder.py:1629), [overlap 정의](/home/dy1013/sirius_multi-gpu/scripts/cache_aware_query_reorder.py:289).

## Future hint와 eviction/admission 구분

runner는 각 쿼리 직전에 `CALL sirius_set_cache_hints('evict;cur=...;next=column:distance,...')`를 보낸다. 거리 단위는 **앞으로 몇 개의 쿼리 뒤에 처음 다시 쓰는가**다. `execution`은 현재 배치의 남은 쿼리만, `run`은 다음 execution까지 본다. 배치 분할 시 현재 배치만 본다. 쿼리별 static all-column 목록으로 만들며 page/RG별 다음 사용 시점이나 필터 선택도까지 계산하지 않는다. [hint 생성](/home/dy1013/sirius_multi-gpu/scripts/future_cache_hints.py:24), [전달](/home/dy1013/sirius_multi-gpu/scripts/run_clickbench_breakdown.py:239), [실제 A hint](/home/dy1013/sirius_multi-gpu/experiment/A_cb_best_r1/celebi_fixed_variable/log_dir/sirius_2026-10-07.log:23).

캐시가 예산을 넘거나 pool 압력을 받으면 page victim 순서는 다음과 같다. hint OFF일 때도 **순수 global LRU가 아니라 STRING page 우선, 그 다음 fixed-width page, 각 그룹 안에서 오래된 page 우선**이다. hint ON이면 **next-use가 먼 page 먼저 → STRING 우선 → 오래된 page 먼저**다. 나중에 다시 쓰지 않는 컬럼은 거리 무한대로 취급한다. 현재 쿼리가 요구하고 이전 hint epoch에서 남아 있던 page는 거리0으로 우선 보호한다. 같은 컬럼의 page에는 같은 거리 힌트가 적용되므로 page 단위 Belady 최적 정책이라고 부르지는 않는다. `SIRIUS_PAGE_EVICT_ORDER=size`를 켜면 STRING 우선 분류 대신 측정한 전체 컬럼 크기 > 예산을 먼저 내보내는 분류를 쓴다. [정렬](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:5547), [victim key](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:5574), [거리](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:5900).

`evict`는 **이미 들어온 page 중 무엇을 내보낼지** 바꾼다. 미래에 안 쓰는 컬럼의 삽입 자체를 금지하지 않는다. `admit`는 다시 사용하지 않는 컬럼의 복사를 생략하고, `bypass`는 store가 가득 찼을 때 모든 resident page보다 다음 사용이 먼 incoming 컬럼을 생략한다. 현재 A best/HIT ro_evict는 `admit`/`bypass`가 아니다. 이와 별도로 공통 admission은 pool free가6GiB floor 밑이면 적재를 거부하거나 headroom에서 중단한다. pressure eviction은 effective cache budget을 줄여 query 작업 공간을 돌려준다. [미래 admission](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:5021), [pool headroom](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:4967), [적재 중단](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:5310), [압력 예산](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:5502).

Big-prefix/oversized skip/replace-admission은 별도 실험 옵션이며 기본 OFF다. **reorder가 큰 컬럼 쿼리를 뒤로 보내는 것**과 **cache가 그 컬럼의 삽입을 거부하는 것**을 같은 정책으로 설명하면 안 된다. [prefix 기본값](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:103), [skip 기본값](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:317), [replace 기본값](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:132).

## 실제 순서와 전체 43개로 확장할 때

TH 기존 arrival과 A/HIT 실제 reordered 순서: [runner](/home/dy1013/sirius_multi-gpu/experiment/A_th_best_r1/runner.log:1).

```text
arrival: 8,1,4,7,17,5,6,2,10,16,12,19,21,14,18,15,22,9,13,20,3,11
reorder: 8,5,7,15,14,6,17,20,11,2,16,9,18,22,13,3,10,4,21,12,1,19
```

CB 기존41개는 자연 순서1~43에서28/29를 제외한다. A/HIT 실제 reordered 순서: [runner](/home/dy1013/sirius_multi-gpu/experiment/A_cb_best_r1/runner.log:3).

```text
32,33,36,31,15,6,13,26,14,17,18,4,5,16,20,9,10,3,2,8,19,25,27,43,7,42,41,11,12,1,30,21,22,24,34,35,37,39,23,40,38
```

**전체43개**를 같은 정책/metadata/6GiB 예산으로 정적 재계산하면 다음과 같다. 벤치마크 실행 결과가 아니라 새 실험의 예상 순서다.

```text
2,8,3,30,10,9,4,5,16,20,11,12,14,17,18,6,13,26,15,31,36,33,32,25,27,19,43,7,42,41,1,21,22,24,28,34,35,37,39,23,40,29,38
```

q28은 `CounterID,URL`, q29는 `Referer`를 요구한다. 이 둘을 포함하면 전체 후보 경로의 점수가 변해 **시작 쿼리도q32→q2로 바뀐다**. 기존41 순서에 두 쿼리를 덧붙인 실험이 아니다. 새 full43 graph는 전체 workload와 재정렬 순서가 달라진 조건임을 명시해야 한다. [컬럼 목록](/home/dy1013/sirius_multi-gpu/test/tpch_performance/clickbench_pin_columns.py:36).

소스의 오래된 “URL row group 3.7GiB”, “큰 컬럼은 served0” 주석은 수정된 실제 page 크기/후속 실험과 맞지 않는다. 여기서는 실행하는 분기·정렬식·저장 launcher·실제 순서를 근거로 설명했고 그 주석의 옛 숫자를 성능 원인으로 사용하지 않았다.
