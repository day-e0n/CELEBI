**결론.** 현재 확인한 결과에서 workload 전체의 cache hit가 100%인 것은 아니다. 일부 ClickBench 쿼리의 scan에 대해서 byte hit 또는 full-batch hit가 100%다. 이는 그 범위의 입력을 GPU 캐시가 공급했다는 뜻이며, 저장한 캐시 데이터가 모두 재사용됐다는 뜻은 아니다.

**현재 성능과 hit 계측은 별도 실험이다.** A_* v3는 2026-10-07의 late materialization 기본 ON·첫 scan 예약 추정 3배 코드로 측정한 결과다. best는 query reorder와 미래 인지 축출을 함께 포함한다. HIT_*는 2026-10-06의 조건당 한 run, execution 2·3을 계측한 결과다. HIT 숫자를 A latency와 같은 실행의 값으로 결합하지 않는다.

| A_* v3 쿼리 wall 합 평균 | baseline | paging | reorder + 미래 인지 축출 |
|---|---:|---:|---:|
| ClickBench | 30.58175s | 26.22025s | 24.03850s (−21.40%) |
| TPC-H SF50 | 32.63233s | 30.67967s | 23.55083s (−27.83%) |

각 run에 전체 execution 1–3의 전체 쿼리 완료가 존재함을 직접 확인했다. ClickBench 2 runs × execution 2·3, TPC-H 3 runs × execution 2·3. [현재 그래프의 입력과 조건](/home/dy1013/sirius_multi-gpu/result/plot_headline_cache_reorder_v3.py:1).

**Byte hit의 정의.** `1 − Σread_bytes / Σtotal_bytes`. total은 조립한 materialized 컬럼 값의 크기, read는 그 중 reader가 생성한 값의 크기다. 실제 함수는 fixed-width 값과 문자열 chars + 32비트 offsets를 세며 null mask·GPU 페이지 allocation padding을 제외한다. NVMe에서 읽은 압축 바이트나 전체 dataset 파일 크기가 아니다. 문자열 부분 범위별 offsets 끝 항목은 read 쪽에서 반복 계산되며, 로그에는 read를 total로 cap하는 코드가 있다. 따라서 이 지표는 현재 계측 정의의 decoded 값 바이트 기준 coverage로 명시한다. [바이트 계산](/home/dy1013/sirius_multi-gpu/src/op/scan/parquet_gpu_ingestible.cpp:189), [최종 로그](/home/dy1013/sirius_multi-gpu/src/op/scan/parquet_gpu_ingestible.cpp:2310), [집계](/home/dy1013/sirius_multi-gpu/result/cache_hit_rate.py:69).

| HIT_* 전체 decoded-byte hit | paging | reorder + LRU | reorder + 미래 인지 축출 |
|---|---:|---:|---:|
| TPC-H SF50 | 8.29% | 13.79% | 26.64% |
| ClickBench | 15.13% | 18.46% | 17.06% |

[전체 CSV](/home/dy1013/sirius_multi-gpu/result/cache_hit_rate.csv). 현재 cache-hit plot의 `paging + reorder`는 ro_evict라 미래 인지 축출을 포함한다. 순수 reorder는 CSV의 `paging + reorder (LRU eviction)` 행이다. 그림 조건명을 구분해야 한다.

**100%에서도 비용이 남는 직접 사례.** 같은 HIT 측정에서:

| ClickBench | cache 조건 | byte / batch hit | baseline wall | cache 조건 wall |
|---|---|---:|---:|---:|
| q25 | reorder + 미래 인지 축출 | 100% / 100% | 0.527s | 0.081s |
| q14 | paging | 100% / 100% | 1.407s | 1.119s |
| q26 | reorder + 미래 인지 축출 | 100% / 100% | 1.063s | 0.7735s |

q14는 SearchPhrase로 GROUP BY하고 UserID의 COUNT DISTINCT를 계산하며 ORDER BY/LIMIT를 수행한다. q26은 문자열 SearchPhrase로 정렬한다. 읽기를 생략해도 GPU copy/concat, post-filter, 집계·정렬, 예약·스케줄링과 결과 반환은 남는다. q14 cache 조립 task 합 0.1239s, post-filter task 합 0.1109s, 읽기 task 합 약 10µs다. 이 task 시간들은 겹치므로 wall에서 직접 뺄 수 없다. [쿼리별 hit](/home/dy1013/sirius_multi-gpu/result/cache_hit_rate_per_query.csv), [SQL](/home/dy1013/sirius_multi-gpu/test/tpch_performance/clickbench_queries.py:22), [hit 복사·concat](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:6376).

**작은 쿼리의 100%를 평균하면 큰 miss가 가려진다.** HIT의 ClickBench q24 요청량은 65.35GiB로 전체 165.97GiB의 약 39.4%이며, reorder+LRU hit는 0.73%다. 작은 입력 쿼리를 완전 hit시켜도 큰 miss가 전체 입력의 대부분을 차지할 수 있다. 현재 A_*에서도 q24 wall은 best 평균 6.19525s로 전체 24.03850s의 25.77%다. q23도 3.014s여서 둘의 비중은 약 38.3%다. HIT의 큰 decoded volume을 그대로 현재 A q24의 원인으로 적용하지 않는다.

**추가 비용과 효과의 상쇄.** 캐시 적재는 GPU D2D 복사를 하고 메모리를 사용한다. A_*의 populate task 합은 TPC-H paging 4.497s, best 2.986s, ClickBench paging 1.281s, best 1.254s다. A ClickBench 메모리 압력 로그는 execution당 paging 13회, best 10.75회이며 TPC-H에서는 해당 pressure 로그가 없다. 메모리 압력을 모든 workload의 원인으로 일반화하지 않는다. 반복 page 재사용량과 critical-path 시간은 별도 계측으로 연결해야 한다.

**현재 q24의 별도 재읽기 경로에서 확인된 병목과 계측 누락.** A_cb_best_r1 execution 2의 q24는 좁은 EventTime/URL scan, TOP_N 이후 선택한 행 번호를 이용해 전체 컬럼을 다시 읽는다. `read_row_numbers()`는 skip_rows/num_rows를 설정한 cuDF read_parquet를 직접 호출한다. 이 경로는 일반 page cache lookup과 공통 scan-audit/cache-hit/populate 코드보다 앞에서 반환한다. 따라서 cache-hit trace를 새 A 코드에서 켜더라도 이 재읽기는 현재 분모와 reader 바이트 집계에서 누락된다. 전체 쿼리의 coverage라고 주장하려면 경로 계측을 먼저 보완해야 한다. [early return](/home/dy1013/sirius_multi-gpu/src/op/scan/parquet_gpu_ingestible.cpp:1760), [별도 reader](/home/dy1013/sirius_multi-gpu/src/op/scan/parquet_gpu_ingestible.cpp:679).

같은 실행의 log에서 재읽기 GPU_SCAN이 4,260,131,360 bytes(약 3.97GiB)를 요청하며 OOM에 도달하고, 6GiB page cache를 전부 drop한 뒤 task 805를 재시도한다. 비어 있는 split은 수 ms지만, 데이터를 읽는 split의 materialize는 약 1.37–4.86s다. q24 wall은 6.267s다. 최종 query-end cache residency는 0이다. [OOM·drop·retry·scan-stage 로그](/home/dy1013/sirius_multi-gpu/experiment/A_cb_best_r1/celebi_fixed_variable/log_dir/sirius_2026-10-07.log:17137). 10개의 행을 반환했다는 로그가 적은 물리 I/O나 작은 decode 작업 메모리를 보장하지는 않는다. 어떤 압축 단위를 실제로 몇 번 읽고 디코딩했는지는 추가 byte/time 계측 대상이다.

**과거 hit 지표와 혼용하지 않는다.** `plot_cache_hit_rate_eligible.py`는 dynamic_filter_scan을 분모에서 제외한다. 그 지표에서 100%는 cache-eligible 대상으로 제한된 값이다. page-weighted 버전은 일부 miss를 uncompressed 추정량/16MiB로 환산한다. 최신 HIT decoded 바이트 coverage와 같은 정의가 아니다. [제외 조건](/home/dy1013/sirius_multi-gpu/scripts/plot_cache_hit_rate_eligible.py:47), [가중치의 근사](/home/dy1013/sirius_multi-gpu/scripts/plot_cache_hit_rate_eligible.py:73).

다음으로 같은 코드·설정·workload에서 쿼리별 decoded-byte coverage, full/partial/miss 비중과 latency를 함께 수집한다. 먼저 q24 late-refetch 경로의 실제 I/O, cuDF reader 시간, 행 번호 대기, OOM 재시도 및 cache drop을 좁은 첫 scan과 분리해 계측한다. 그 다음 A_*의 q23 및 full-hit 상태의 q14·q26을 비교한다. byte trace가 문자열 크기 조회로 동기화할 수 있으므로 성능용 실행과 계측용 실행을 구분한다. 캐시에 넣었던 페이지 중 실제로 다시 사용된 비율은 별도 page-lifetime 지표다. 새 실험이나 엔진 변경은 이번 검토에서 수행하지 않았다.
