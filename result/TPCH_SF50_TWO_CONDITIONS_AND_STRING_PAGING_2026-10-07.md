# TPC-H SF50 두 조건 결과와 STRING 페이지 구현

2026-10-07 확인. 비교 대상은 **baseline 대 reorder_future**다.
CELEBI 조건은 decoded page caching + query reordering + future-aware eviction이다.
`reorder_lru`는 이 문서와 최종 그래프의 비교 대상에서 제외했다.
완료된 데이터를 재분석했으며 엔진이나 진행 중인 실험 설정은 변경하지 않았다.

## 완료 여부와 측정 방법

원본 디렉터리:

```text
experiment/FULL_CACHE_PIN_20261007/runs/latency/th_r1/baseline
experiment/FULL_CACHE_PIN_20261007/runs/latency/th_r1/reorder_future
```

각 조건은 q1–q22를 세 번 실행하여 **66/66 쿼리가 성공**했다.
metadata의 `status=complete`, `comparison_valid=true`, return code 0을 확인했다.
CELEBI 실행 로그에는 실제 `CALL sirius_set_cache_hints('evict;...')`가 66회 있다.
첫 execution 전체는 warmup으로 제외하고 execution 2·3을 평균했다.
독립적인 프로세스 실행은 조건당 한 번이며, 쿼리당 평균 표본은 두 개다.
이 결과는 여러 독립 run의 평균이나 통계적 유의성을 입증하는 결과로 표현하지 않는다.

| 지표: 22개 쿼리 평균의 합 | Baseline | CELEBI | 감소율 |
|---|---:|---:|---:|
| 쿼리 실행 경과 시간 합 | 32.5629 s | 23.8170 s | 26.86% |
| scan operator work 합 | 104.4389 s | 73.3382 s | 29.78% |
| 모든 operator work 합 | 109.6186 s | 78.4331 s | 28.45% |

쿼리 경과 시간은 Python의 `execute().fetchall()` 전후를 측정한다.
쿼리 label 설정, future hint 전달, `SET gpu_execution`, 순서 생성은 타이머 밖에 있다.
따라서 위 32.56 s와 23.82 s는 **쿼리 실행 시간의 합**이며 hint 준비를 포함하는 배치 전체 경과 시간은 아니다.

Operator work는 Quent telemetry의 task가 `Computing` 상태에 머문 시간을
해당 `current_operator_id`에 귀속하여 모든 task에 걸쳐 더한 값이다.
CPU 작업, I/O 대기 및 동기화가 그 상태 안에서 발생하면 포함된다.
여러 task의 시간이 겹치므로 합계가 쿼리 경과 시간보다 클 수 있다.
CUDA kernel active time이나 wall-clock latency의 배타적인 분할로 해석하면 안 된다.
파서와 bucket 매핑: [parse_quent_operator_breakdown.py](/home/dy1013/sirius_multi-gpu/scripts/parse_quent_operator_breakdown.py:1).

대부분의 감소는 scan에 귀속된다. join work 합은 2.5479 s → 2.5692 s로 거의 같고,
aggregate work 합은 1.4950 s → 1.4559 s다.
q1의 쿼리 시간은 1.5068 s → 1.5434 s로 약 2.4% 증가한다.
모든 쿼리가 빨라졌다고 표현하지 않는다.
이 두 조건 비교만으로 paging, reorder, future eviction 각각의 기여를 분리할 수는 없다.

## 실험 환경과 정책

| 항목 | 확인된 설정 |
|---|---|
| 입력 | `/mnt/nvme/sirius_tpch/tpch_parquet_sf50_optimized`, TPC-H SF50 q1–q22 |
| GPU | `CUDA_VISIBLE_DEVICES=1`, RTX A5000 한 장, 보드 메모리 약 24 GiB |
| 노드 CPU | Intel Xeon w7-3445, 20 physical cores / 40 logical CPUs |
| GPU driver | 595.71.05, 이번 확인 시 `nvidia-smi` 출력 |
| GPU memory YAML | `usage_limit_bytes: 20GB`, reservation fraction 0.85 |
| HOST memory YAML | `capacity_bytes: 32GB` |
| Decoded page cache 예산 | GPU당 4 GiB, flat store의 공통 예산 |
| Cache page allocation | 16 MiB = 16,777,216 bytes, fixed-width/STRING 모두 |
| Free-memory floor | GPU당 6 GiB |
| Late materialization | ON |
| Prefetch cache | OFF |
| O_DIRECT | ON |
| Telemetry | Quent ON; per-materialization cache-hit trace OFF |
| CELEBI reorder | `small-first-big-last` |
| CELEBI future hints | `mode=evict`, `horizon=execution` |

GPU/HOST YAML의 `GB`를 확인 없이 `GiB`로 바꾸지 않는다.
실행 harness metadata의 GPU/HOST byte 필드는 binary 단위를 가정하지만,
그 숫자만으로 실제 allocator 용량을 확정하지 않았다. Cache 예산과 page 크기는 명시적인 byte 설정이다.
노드에 GPU 네 장이 있지만 이 비교는 한 장만 사용한 결과다.

Baseline은 도착 순서를 사용하고 GPU page reuse/populate가 OFF다.
CELEBI는 fixed-width와 STRING page caching 모두 ON이며 아래 순서로 SQL을 실행한다.
Reorder는 SQL 실행 순서를 바꾸며 각 SQL의 join/aggregate/sort 연산 순서를 직접 바꾸는 기능이 아니다.

```text
baseline: 8,1,4,7,17,5,6,2,10,16,12,19,21,14,18,15,22,9,13,20,3,11
CELEBI:   8,5,7,15,14,6,17,20,11,2,16,9,18,22,13,3,10,4,21,12,1,19
```

Future hints는 현재 execution의 **남은 쿼리 중 각 컬럼의 첫 다음 사용까지의 쿼리 수**를 알려준다.
다음 execution까지 들여다보지 않으며 page별/row-group별 접근 예측도 아니다.
Eviction은 다음 사용이 가장 먼 컬럼의 page부터 선택하고, 다시 사용하지 않는 컬럼을 가장 먼저 고려한다.
동점은 STRING 우선, 그 다음 오래된 page 순이다.
`evict` 모드는 미래에 안 쓰는 컬럼의 admission 자체를 금지하는 `admit` 모드와 구분한다.
Page별 Belady 최적 정책이라고 부르지 않는다.
[hint 생성](/home/dy1013/sirius_multi-gpu/scripts/future_cache_hints.py:24),
[실제 hint 전달](/home/dy1013/sirius_multi-gpu/scripts/run_full_cache_pin_comparison.py:384),
[eviction 정렬](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:5574).

같은 extension build를 사용했다. SHA-256:
`0b53e79926ac5457625b95c146aef350ac5b4d74978b302e3278303098e79a19`.
원본 metadata, telemetry 파일 hash 및 실제 hint 호출 위치는
[provenance JSON](/home/dy1013/sirius_multi-gpu/result/tpch_sf50_two_conditions_provenance.json)에 저장했다.

## STRING 페이지: 어느 단계가 CPU이고 GPU인가

여기서 page는 **decoded GPU cache page**다. Parquet encoded page나 OS page를 뜻하지 않는다.
설명 대상은 이 실험의 flat page store에서 처리하는 cuDF `STRING` 컬럼이다.
LIST/STRUCT 등 모든 variable-width 타입에 같은 구현이 적용된다고 일반화하지 않는다.

```text
Parquet의 선택된 scan batch
  → GPU decode: chars + offsets + null mask
  → offsets metadata를 GPU에서 HOST로 복사
  → CPU에서 실제 byte footprint를 계산하며 binary search
  → GPU에서 페이지별 offsets rebasing 및 D2D copy
  → 독립된 16 MiB GPU cache pages
```

전체 테이블을 먼저 decode하는 것이 아니라 해당 scan이 읽은 batch를 대상으로 한다.
`read_parquet(...).tbl` 이후 cache population을 호출한다.
여러 CUDA 작업이 비동기로 제출되더라도 offsets D2H 뒤 stream synchronization으로
CPU 탐색 전에 필요한 decode/offsets 작업의 완료를 기다린다.
[GPU reader](/home/dy1013/sirius_multi-gpu/src/op/scan/parquet_gpu_ingestible.cpp:2230),
[population 호출](/home/dy1013/sirius_multi-gpu/src/op/scan/parquet_gpu_ingestible.cpp:2350).

### 1. Offsets는 이미 누적 문자열 바이트 위치다

R행 cuDF STRING 컬럼은 chars와 R+1개의 offsets를 갖는다.
행 i의 문자열 byte 길이는 `o[i+1] - o[i]`, 연속 구간 `[a,b)`의 chars byte 수는 `o[b] - o[a]`다.
UTF-8 문자의 개수가 아니라 **byte 수**이며, 평균 문자열 길이를 추정할 필요가 없다.
Offsets 원소 폭 w는 실제 타입에 따라 4 또는 8 bytes다.

해당 `(file, column, row_group, row_count)`에 저장된 page layout이 없을 때,
그 row group의 R+1 offsets를 HOST로 복사한다.
INT32 offsets는 복사 후 CPU의 INT64 누적 위치 배열로 변환하고 INT64도 지원한다.
페이지 경계를 찾기 위해 chars를 CPU로 복사하지 않는다.
[layout lookup와 D2H](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:5107).

### 2. 페이지에 필요한 전체 footprint를 계산한다

페이지 예산을 P=16 MiB, 구간의 행 수를 n=b−a라고 하자.
`A(x)=256×ceil(x/256)`이며 `M(n)`은 cuDF가 요구하는 n행 null-mask allocation 크기다.
`M(n)`은 cuDF 자체 padding을 포함하므로 단순한 `ceil(n/8)`과 항상 같지는 않다.

```text
need(a,b) = A(o[b] − o[a]) + A((b−a+1) × w) + A(M(b−a))
b*        = max { b : b>a and need(a,b) ≤ P }
```

Chars, offsets, mask 영역마다 256-byte alignment를 고려한다.
Chars만 16 MiB까지 채우는 방식이면 나머지 metadata 때문에 전체 allocation이 page 예산을 초과한다.
경계 계산에서는 **항상 mask가 존재한다고 가정**하여 scan마다 nullability가 달라도
동일 row group의 page grid를 유지한다.
실제 row group에 null이 없으면 mask를 저장하지 않아 해당 여유 공간도 padding이 된다.
[footprint 함수](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:4744),
[mask-inclusive 경계](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:5073).

### 3. CPU binary search로 마지막 행을 결정한다

`need(a,b)`가 b에 대해 단조 증가하므로 CPU에서 binary search하여 가장 큰 b를 찾는다.
그 b를 다음 페이지의 시작 행으로 삼아 row group 끝까지 반복한다.
한 STRING 값을 중간에서 나누지 않고, 페이지가 row-group 경계를 넘지 않도록 한다.
평균 길이에 따른 고정 행 수 분할이나 GPU에서 수행하는 binary search가 아니다.
[실제 탐색 loop](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:5143).

```text
a = 0
while a < R:
    if need(a,R) <= P: emit [a,R); break
    b = binary_search_largest_fitting_end(a,R)
    if b == a: do not cache this column's entire row group; stop
    emit [a,b)
    a = b
```

위 의사코드의 emit은 경계 기록이다. oversized row가 발견되면 기록한 layout 전체를 버린다.
**한 행조차 offsets/mask와 함께 page에 들어가지 않으면 해당 컬럼의 RG 전체를 cache하지 않는다.**
해당 값은 원래 reader 출력으로 쿼리에서 사용하며 큰 page를 새로 만들거나 값을 여러 page로 쪼개지 않는다.
같은 RG의 다른 컬럼도 함께 cache 불가능해지는 것은 아니다.
페이지 행 수는 문자열 길이 분포에 따라 달라지고 컬럼별 page grid도 서로 다를 수 있다.

### 4. GPU에서 독립적인 페이지를 만든다

각 구간의 STRING offsets에서 첫 chars offset을 빼서 페이지 내부 기준 0으로 만든다.
이 rebasing은 `cudf::binary_operation(..., SUB)`로 GPU에서 수행한다.
Chars는 원래 decoded column에서 GPU page slab으로 D2D copy한다.
Null mask가 필요하면 시작 행의 bit가 0이 되도록 slice/copy하고 rebased offsets도 slab으로 D2D copy한다.

기본 설정의 page는 **한 개의 16 MiB `rmm::device_buffer`**이며 물리 배치는
`chars → null mask → rebased offsets`다. 각 영역 시작은 256-byte aligned다.
컬럼 view와 offsets child view가 이 slab을 참조한다.
최초 population은 별도 page allocation과 데이터 복사가 필요하며 zero-copy가 아니다.
[GPU rebasing](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:4833),
[allocation과 D2D](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:4867).

기본값에서는 RG의 마지막 page도 16 MiB라서 사용하지 않는 tail은 내부 padding이다.
별도 `SIRIUS_PAGE_TAIL_SHRINK` 옵션은 기본 OFF다.
Cache key는 `(file, column, row_group, page_index)`이며 `row_offset`과 `num_rows`도 기록한다.
저장된 실제 page bytes는 chars payload뿐 아니라 mask, offsets 및 slab padding까지 포함한다.

### 5. 재사용과 비용의 범위

계산된 행 경계는 `_page_layouts`에 저장해 다음 population에서 재사용한다.
이 경우 전체 R+1 offsets D2H와 CPU binary search를 반복하지 않는다.
하지만 page 생성에 필요한 p+1개 경계 위치의 chars offsets를 읽는 D2H와 stream synchronization은 남는다.
문자열 값 자체를 캐시하지 못했거나 evict했더라도 **경계 metadata의 재사용과 데이터 cache hit는 별개**다.
[경계 offsets의 D2H](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:5260).

R행, p페이지에 대해 첫 layout 계산은 offsets 전송 O(Rw), CPU 탐색 O(p log R)이다.
전체 population은 여기에 GPU 데이터 복사와 offsets rebasing 비용을 더하므로 전체 비용을 O(log R)라고 쓰면 안 된다.
원본 decoded batch가 쿼리에 필요하고 신규 cache pages도 별도 allocation이므로
population 중에는 원본과 cache page, offsets/mask temporary가 동시에 살아 있을 수 있다.
GPU headroom과 cache 예산에 따른 admission 거부/중단도 있으므로 모든 decoded 데이터가 반드시 저장되지는 않는다.
이 방식은 **첫 Parquet decode 자체의 메모리를 줄이는 기법이 아니라 이후 재사용·eviction의 단위를 만드는 방식**이다.
이 비용이 실제 성능 병목인지 판단하려면 별도 계측이 필요하다.

별도 pinned-entry cache의 `variable_width_page_index.cpp`에는 다른 splitter와 fallback이 남아 있다.
이 실험의 flat store 설명과 구분해야 하며 “Sirius의 모든 variable-width 경로가 동일하다”라고 쓰지 않는다.
기존 전체 경로 감사: [VARIABLE_WIDTH_PAGES_2026-10-07.md](/home/dy1013/sirius_multi-gpu/result/VARIABLE_WIDTH_PAGES_2026-10-07.md).

## 논문에 사용할 영문 설명

> For variable-width STRING columns, Sirius constructs decoded cache pages after GPU Parquet decoding. For each eligible column and row group, it transfers the decoded offsets array to host memory and uses CPU binary search to select the largest contiguous row interval whose character payload, offsets, and conservatively provisioned null mask fit within a 16 MiB page, including 256-byte alignment of each region. Page boundaries follow actual byte offsets rather than average string lengths and do not split individual values or cross row-group boundaries. Each page is materialized in a separate GPU allocation through device-to-device copies, with offsets rebased to zero on the GPU. The row boundaries are retained for subsequent cache population, avoiding repeated transfer of the full offsets array and boundary search; boundary-offset reads and page-copy costs remain. The number of rows varies across pages, while the default allocation size remains fixed, including the final page of each row group.

권장 figure caption:

> Per-query operator work breakdown for TPC-H SF50: baseline versus decoded page caching with query reordering and future-aware eviction (CELEBI). Bars show sums of task Computing-state durations attributed to each operator class, averaged over executions 2 and 3 after one warmup execution. Operator work includes overlapping task intervals and is not wall-clock query latency.

## 결과 파일과 재생성

- [PNG](/home/dy1013/sirius_multi-gpu/result/tpch_sf50_two_conditions_operator_breakdown.png)
- [SVG](/home/dy1013/sirius_multi-gpu/result/tpch_sf50_two_conditions_operator_breakdown.svg)
- [PDF](/home/dy1013/sirius_multi-gpu/result/tpch_sf50_two_conditions_operator_breakdown.pdf)
- [쿼리별 평균 CSV](/home/dy1013/sirius_multi-gpu/result/tpch_sf50_two_conditions_operator_breakdown.csv)
- [원본 표본 CSV](/home/dy1013/sirius_multi-gpu/result/tpch_sf50_two_conditions_samples.csv)

```bash
pixi run -e duckdb-python python result/analyze_tpch_sf50_two_conditions.py
pixi run -e duckdb-python python result/plot_tpch_sf50_operator_breakdown.py \
  --input result/tpch_sf50_two_conditions_operator_breakdown.csv \
  --output-prefix result/tpch_sf50_two_conditions_operator_breakdown \
  --proposed-condition reorder_future \
  --proposed-label 'Page caching + reordering + future-aware eviction (CELEBI)' \
  --expected-samples 2
```
