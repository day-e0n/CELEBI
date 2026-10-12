# 히트율 그래프의 데이터와 분모

**표현 수정:** 아래 기존 쿼리별 byte/full-batch 그림은 사용자가 의도한 조건부 100% 지표가 아니다.
Workload별 조건부 지표는 최신 전체 쿼리 로그로 별도 생성했다.
[조건부 캐시 공급 완료율](/home/dy1013/sirius_multi-gpu/result/CONDITIONAL_CACHE_SUPPLY_2026-10-07.md).

이번에 생성한 `cache_hit_requested_legacy*`는 완료된 기존 HIT 로그의 재집계다.
직전 operator breakdown의 `FULL_CACHE_PIN_20261007/runs/latency` 실행과 같은 실행의 수치가 아니다.
그 latency 실행은 `SIRIUS_CACHE_HIT_TRACE=0`이고 `[cache-hit]` 로그가 없었다.
새 full-workload hit 계측이 완료되면 `cache_hit_requested_full*`를 별도로 생성한다.
두 데이터의 코드 버전, 계측 범위 및 query set을 섞지 않는다.

## 현재 확인 가능한 결과

| 기존 HIT 로그 | paging | paging + reorder + future-aware eviction |
|---|---:|---:|
| TPC-H SF50, q1–q22 | 8.2911% | 26.6415% |
| ClickBench, 41개 쿼리 | 15.1283% | 17.0556% |

Source directories:

```text
experiment/HIT_th_lru
experiment/HIT_th_ro_evict
experiment/HIT_cb_lru
experiment/HIT_cb_ro_evict
```

Runner의 모든 query 완료 기록을 검증했다. TPC-H는 조건당 22×3개, ClickBench는 41×3개다.
execution 1은 warmup으로 제외하고 execution 2·3의 바이트를 합산했다.
조건당 독립 run은 한 번이다. 기존 ClickBench는 **q24를 포함하며 q28/q29가 없다.**
쿼리별 그림에는 q1–q43 위치를 유지하고 q28/q29를 N/A로 표시했다.
로그의 label prefix는 모든 조건에서 `paging`이므로 조건 구분은 디렉터리를 기준으로 한다.

## 두 가지 분모

### Byte-weighted hit

```text
hit = Σ(total_bytes − read_bytes) / Σtotal_bytes × 100
```

계측된 materialization에서 공급한 decoded 입력 바이트 중 reader가 만들지 않은 비율이다.
쿼리별 hit의 산술평균이 아니라 **요청 바이트로 가중한 비율**이다.
같은 값이 여러 쿼리에서 반복 요청되면 매번 분모에 들어간다.
따라서 고유 dataset 중 cache에 담긴 비율, cache occupancy, 캐시에 넣은 값이 다시 쓰인 비율,
압축된 NVMe I/O 절약률을 뜻하지 않는다.

이번 legacy 그래프의 분모는 기존 `[cache-hit]`가 기록한 **일반 Parquet materialization 경로**다.
기존 계산에서는 STRING offsets가 포함되며 null mask 및 allocation padding은 제외된다.
Late-materialization refetch와 cached-provider 경로의 누락이 있으므로
이 legacy 값을 SQL의 모든 읽기 경로에 대한 coverage로 표현하지 않는다.
그림 y축을 `Cache hit ratio (% of traced bytes)`로 표시했다.

### Full-batch hit

```text
hit = 전체 입력을 캐시로 공급한 nonempty batch 수
      / 계측된 모든 nonempty batch 수 × 100
```

배치 크기와 무관하게 각 batch를 한 번씩 센다.
일부 column/row만 cache에서 공급하는 splice/partial은 full-batch hit에 포함하지 않는다.
0-byte batch는 분모에서 제외한다. byte denominator가 0이면 N/A이며 0%/100%로 만들지 않는다.
아주 작은 batch의 hit와 큰 batch의 miss가 공존할 수 있으므로 byte-weighted hit와 값이 다를 수 있다.

## 실제 100%와 큰 miss의 사례

아래 바이트는 execution 2·3의 합이며 CELEBI 조건이다.

| ClickBench | 계측된 total bytes | reader bytes | byte hit | full-batch hit |
|---|---:|---:|---:|---:|
| q24 | 140,341,402,440 | 139,755,066,878 | 0.4178% | 0% |
| q25 | 3,879,212,652 | 0 | 100% | 100% |
| q26 | 2,279,252,684 | 0 | 100% | 100% |

q25/q26의 100%는 **그 쿼리의 계측된 scan 입력이 모두 캐시에서 공급됐다**는 뜻이다.
전체 workload가 100%라는 뜻도, GPU copy/aggregation/sort가 사라진다는 뜻도 아니다.
q24만 전체 traced total bytes의 약39.4%를 차지하므로 여러 작은 쿼리의 100%에도
전체 workload의 weighted hit는 17.1%에 머물 수 있다.

이 그림은 값을 100%로 만들기 위해 miss를 임의로 제외하지 않았다.
예전 `cache-eligible hit`처럼 dynamic-filter scan을 분모에서 제외하는 지표는 별도 정의다.
그 지표를 사용한다면 제외한 경로와 양을 함께 보고해야 한다.

## 새 full-workload 결과 생성

`plot_requested_cache_hits.py --source full --watch` 후처리 프로세스가 실행 중이다.
진행 중인 벤치마크 설정이나 query order를 수정하지 않고 metadata만 확인한다.
다음 hit 조건 네 개가 모두 완료되고 모든 쿼리가 성공한 경우 로그를 검증한 뒤 그림을 생성한다.

```text
FULL_CACHE_PIN_20261007/runs/hit/th_r1/paging_lru
FULL_CACHE_PIN_20261007/runs/hit/th_r1/reorder_future
FULL_CACHE_PIN_20261007/runs/hit/cb_r1/paging_lru
FULL_CACHE_PIN_20261007/runs/hit/cb_r1/reorder_future
```

새 결과는 TPC-H q1–q22, ClickBench q1–q43 전부다.
`metric=value_bytes_v2`의 parquet/cached_provider/late_refetch 세 경로를 포함한다.
분모는 fixed-width value data + STRING chars이며 offsets, masks, allocation padding은 제외한다.
예전 legacy trace와 합산하지 않는다. 같은 SQL의 materialization 재시도가 기록된 경우에는
그 attempt의 공급량도 계측량에 포함될 수 있다.
케이스가 실패하거나 query trace가 부족하면 오류 상태를 기록하고 불완전한 결과를 0%로 그리지 않는다.

실행 정보와 상태:

- [watcher 정보](/home/dy1013/sirius_multi-gpu/result/cache_hit_requested_full_watcher.json)
- [상태 JSON](/home/dy1013/sirius_multi-gpu/result/cache_hit_requested_full_status.json)
- [watcher 로그](/home/dy1013/sirius_multi-gpu/result/cache_hit_requested_full_watcher.log)

완료 시 파일 이름은 `result/cache_hit_requested_full{,_per_query}.{png,svg,pdf}`다.
CSV 및 source hash provenance도 같은 prefix로 저장한다.

## 현재 생성된 파일

- [전체 byte hit PNG](/home/dy1013/sirius_multi-gpu/result/cache_hit_requested_legacy.png)
- [전체 byte hit SVG](/home/dy1013/sirius_multi-gpu/result/cache_hit_requested_legacy.svg)
- [전체 byte hit PDF](/home/dy1013/sirius_multi-gpu/result/cache_hit_requested_legacy.pdf)
- [쿼리별 byte/full-batch hit PNG](/home/dy1013/sirius_multi-gpu/result/cache_hit_requested_legacy_per_query.png)
- [쿼리별 SVG](/home/dy1013/sirius_multi-gpu/result/cache_hit_requested_legacy_per_query.svg)
- [쿼리별 PDF](/home/dy1013/sirius_multi-gpu/result/cache_hit_requested_legacy_per_query.pdf)
- [전체 CSV](/home/dy1013/sirius_multi-gpu/result/cache_hit_requested_legacy.csv)
- [쿼리별 CSV](/home/dy1013/sirius_multi-gpu/result/cache_hit_requested_legacy_per_query.csv)
- [Provenance](/home/dy1013/sirius_multi-gpu/result/cache_hit_requested_legacy_provenance.json)

재생성:

```bash
pixi run -e duckdb-python python result/plot_requested_cache_hits.py --source legacy
pixi run -e duckdb-python python result/plot_requested_cache_hits.py --source full
```
