# 조건부 캐시 공급 완료율

사용자가 확인한 지표는 **캐시 경로로 선택된 요청 내부의 공급 완료 비율**이다.
쿼리별 byte hit 또는 전체 materialization 중 full batch가 차지하는 비율과 구분한다.
새 그림은 workload별로 이 조건부 비율을 하나씩 보여준다.

## 최신 전체 쿼리 결과

`FULL_CACHE_PIN_20261007/runs/latency`의 완료된 케이스에서 추출했다.
TPC-H는 q1–q22, ClickBench는 **q1–q43 전체이며 q24/q28/q29도 포함**한다.
각 조건은 세 번 실행했고 execution 1을 제외한 execution 2·3의 이벤트를 합산했다.
모든 query 기록이 성공이며 `status=complete`, `comparison_valid=true`인 경우만 사용했다.

| Workload | 조건 | 경로 선택 | 공급 완료 및 audit | 조건부 완료율 |
|---|---|---:|---:|---:|
| TPC-H SF50 | paging | 24 | 24 | 100% |
| TPC-H SF50 | paging + reorder + future eviction | 67 | 67 | 100% |
| ClickBench | paging | 67 | 67 | 100% |
| ClickBench | paging + reorder + future eviction | 120 | 120 | 100% |

그림은 사용자가 지정한 미래 인지 eviction 포함 조건의 workload별 두 값을 표시한다.
CSV에는 비교용 paging 조건도 보관했다.

## 정확한 분모와 로그 근거

```text
조건부 공급 완료율
  = 공급 완료가 확인된 full-cache materialization batch 수
    / full-cache 경로로 선택된 materialization batch 수 × 100
```

1. `[fixed-page-cache] column_splice cached>0 read=0`: 선택 기록.
2. `[fixed-page-cache] full_hit`: 캐시 입력으로 scan output table이 만들어진 기록.
3. `[scan-audit] parquet_materialize full_hit=1`: 그 materialization의 audit 기록.

단순히 전체 합계가 같다고 판단하지 않고 `(query, execution, file, row_group_count)`별로
선택 수, table 공급 수, audit 수를 대조했다. 완료 수가 선택 수를 초과하면 분석을 거부한다.
분모가 0인 쿼리는 N/A다. 분모 없는 쿼리를 100%나 0%로 채워 평균하지 않는다.

**선택 시점의 의미:** `column_splice`는 cache lookup 시도 전에 남기는 로그가 아니다.
`cached_row_group_columns()`가 실제 페이지를 건네준 뒤 `read_names`를 재구성하여
캐시에서 모든 컬럼을 확보한 경우 full-cache branch를 선택한다.
그 다음 table 생성과 audit을 대조했으므로 이 값은 **선택 후 공급 완료율**이다.
Lookup 시도 전체에 대한 hit 확률이나 cache admission 정책의 효과를 측정한 값은 아니다.

코드:

- [페이지 handover와 경로 선택](/home/dy1013/sirius_multi-gpu/src/op/scan/parquet_gpu_ingestible.cpp:2147)
- [선택 로그](/home/dy1013/sirius_multi-gpu/src/op/scan/parquet_gpu_ingestible.cpp:2181)
- [full-cache table 생성](/home/dy1013/sirius_multi-gpu/src/op/scan/parquet_gpu_ingestible.cpp:2220)
- [분석 스크립트](/home/dy1013/sirius_multi-gpu/result/plot_conditional_cache_supply.py)

현재 로그에서 분리하여 대조할 수 있는 **일반 Parquet 경로의 full-cache batch**로 범위를 한정했다.
일부 컬럼이나 행만 캐시에서 공급하는 혼합 경로, reader miss, late refetch와 별도 cached-provider는
이 조건부 분모에 포함하지 않는다. 따라서 두 workload의 전체 입력 요청이 100% hit한 것이 아니다.
캐시에 저장했던 모든 page가 사용됐다는 뜻도 아니다.
이 기록은 scan 입력 table 공급 단계이고 쿼리의 filter/join/aggregate/sort 전체 완료 단계와도 다르다.

## 전체 요청 기준 비율과 함께 설명하기

미래 인지 eviction 조건의 일반 Parquet materialization audit은 TPC-H 938개,
ClickBench 330개다. 그중 위 조건부 분모에 들어간 것은 각각 67개, 120개다.
부분 캐시 공급도 있을 수 있으므로 이 batch 선택 비율을 byte coverage로 바꾸면 안 된다.

전체 데이터 공급 효과는 별도 `캐시가 공급한 decoded value bytes / 모든 경로의 decoded value bytes`로
계산한다. 최신 `value_bytes_v2`의 전체 쿼리 byte trace 실행은 별도로 진행되고 있다.
이전 HIT 로그의 26.6%/17.1%를 위 최신 실행의 byte coverage 값으로 결합하지 않는다.

발표 문장:

> 캐시에 모든 입력이 있음을 확인해 캐시 경로로 선택한 batch는 두 workload 모두 공급을 완료했습니다.
> 이 조건부 비율은 100%이며, 전체 요청 중 캐시가 공급한 비중은 별도의 byte coverage로 제시합니다.

영문 caption:

> Conditional supply completion for scan batches selected for the full-cache path, with page caching, query reordering, and future-aware eviction. Completion was verified for 67/67 TPC-H SF50 batches and 120/120 ClickBench batches across executions 2 and 3. The denominator is restricted to full-cache selections after page handover; this is not workload-wide cache-hit probability or byte coverage. ClickBench includes all 43 queries.

## 파일

- [PNG](/home/dy1013/sirius_multi-gpu/result/conditional_cache_supply.png)
- [SVG](/home/dy1013/sirius_multi-gpu/result/conditional_cache_supply.svg)
- [PDF](/home/dy1013/sirius_multi-gpu/result/conditional_cache_supply.pdf)
- [CSV](/home/dy1013/sirius_multi-gpu/result/conditional_cache_supply.csv)
- [Provenance](/home/dy1013/sirius_multi-gpu/result/conditional_cache_supply_provenance.json)

```bash
pixi run -e duckdb-python python result/plot_conditional_cache_supply.py
```
