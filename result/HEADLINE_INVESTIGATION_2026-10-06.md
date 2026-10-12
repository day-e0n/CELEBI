첨부 그래프 추가 분석 — 2026-10-06

**범위와 재현.** `headline_cache_reorder_v2`의 `FA_{th,cb}_{base,lru,ro_lru}_r*` 15개 run만 분석했다. 미래 인지 축출 조건은 포함하지 않았다. 모든 run에서 execution 1–3의 전체 쿼리 완료를 확인했다. TH는 SF50/22쿼리/4GiB/worst, CB는 41쿼리(q28·q29 제외)/6GiB/natural, GPU 하나, O_DIRECT, prefetch OFF, min-free 6GiB다. 엔진 코드 변경, 새 벤치마크 실행, 기존 설정 변경은 하지 않았다. 새 Python 스크립트는 기존 로그를 읽고 분석 파일만 생성한다.

**그래프의 숫자.** execution 2·3의 쿼리 wall 합 평균이다. 러너의 재정렬 계산·힌트 설정·초기화·요청을 모으는 대기 시간은 제외한다.

| 벤치마크 | baseline | paging | paging + reorder |
|---|---:|---:|---:|
| ClickBench | 30.8388s | 27.8285s | 26.7690s |
| TPC-H SF50 | 32.3418s | 30.8557s | 29.6525s |

재정렬의 추가 절약은 CB 1.0595s, TH 1.2032s다. 빨라진 쿼리의 절약 합에서 느려진 쿼리의 손실을 뺀 값이다.

| 벤치마크 | 빨라진 쿼리 절약 합 | 느려진 쿼리 손실 합 | 손실로 상쇄된 비율 |
|---|---:|---:|---:|
| ClickBench | 1.8388s | 0.7793s | 42.4% |
| TPC-H SF50 | 1.7155s | 0.5123s | 29.9% |

독립 run별 재정렬 절약은 CB 1.106/1.013s, TH 1.4795/1.425/0.705s다. 같은 run의 execution 2·3은 캐시를 공유하므로 독립 run 두 개와 같게 취급하지 않는다.

**1. 반복 경계와 첫 실행 비용을 먼저 분리한다.** 현재 그래프는 connection과 캐시를 유지하며 동일 batch를 반복한 결과다. TPC-H 첫 execution은 baseline 32.4373s, paging 33.8930s, reorder 33.3017s다. 첫 실행을 포함한 총 3 execution에서는 paging −1.56%, reorder −4.65%이며, 그래프의 반복 실행 효과 −4.6%/−8.3%보다 작다. CB의 전체 3회 효과는 −9.26%/−12.50%다.

TPC-H paging의 read task 합은 첫 실행 88.60s와 반복 실행 88.73s로 거의 같지만, populate task 합은 11.456s에서 3.338s로 줄었다. 첫 실행의 적재 비용을 버리는 것이 결과 해석에 중요한 영향을 준다. 이 타이머는 동시에 도는 scan task 합이며 wall 비용이 아니다. O_DIRECT이므로 이를 OS 파일 캐시가 따뜻해진 효과라고 설명하지 않는다.

특히 q8은 두 캐시 조건에서 모두 첫 번째인데 반복 실행 wall이 2.2862→1.8810s, read task 합이 8.1736→6.5344s로 줄었다. q8의 0.4052s가 재정렬 순이득의 33.7%다. 첫 execution의 q8은 2.565→2.628s로 느려졌다. 원래 순서 끝의 q11과 재정렬 끝의 q19가 다음 q8에 남기는 상태를 비교해야 한다. full-hit batch는 오히려 1→0, partial serving은 0→1.5이므로 full-hit 개수만으로 효과를 판단할 수 없다.

다음 확인: 끝→처음 경계의 page ID·살아 있는 row 범위·다음 사용을 추적한다. 같은 실행 안에서의 재사용과 이전 execution에서 넘어온 재사용을 따로 센다. 진단 조건으로 batch마다 새 프로세스에서 실행한 결과를 비교하되, 각 batch 내부의 캐싱은 유지한다. 기존 결과의 반복 효과와 새로운 단발 batch 결과를 분리해 보고한다.

**2. q24의 메모리 압력이 q25의 재사용을 어떻게 끊는지 추적한다.** CB paging에서 q25는 0.4863s, reordered q25는 0.0810s다. 자연 순서의 q24 종료 후 SearchPhrase live page는 모든 표본에서 0개이며 q25에서 served counter가 증가하지 않는다. 재정렬은 q25를 q24 앞에 두고 5/5 batch를 완전 히트시킨다. q6·q13도 SearchPhrase 완전 히트가 0/3→3/3으로 바뀐다.

반대쪽 손실도 함께 본다. CB q32는 위치 30→1 이동 후 0.4655→0.7620s, partial serving 2.25→0.75 batch다. q14는 1.0873→1.1860s, full hit 5→0, read task 합 약 0→1.009s다. TH 큰 회귀는 q21 +0.1683s, q1 +0.1545s, q9 +0.1297s다. 이 쿼리들에서 새로 얻은 컬럼 재사용과 잃은 컬럼 재사용을 같은 페이지 기준으로 비교한다.

**3. 설정 예산과 실제 유지 가능한 예산의 차이를 확인한다.** cached 조건 모두 메모리 압력으로 eviction의 유효 예산이 순간적으로 0까지 내려간다.

| 벤치마크/조건 | 적재 거부 batch | scan batch | pressure 이벤트 | free 증가가 보이지 않은 이벤트 |
|---|---:|---:|---:|---:|
| TH paging | 44.3 | 469 | 20.7 | 19.8 |
| TH reorder | 40.3 | 469 | 18.7 | 18.2 |
| CB paging | 58.3 | 156 | 22.0 | 21.0 |
| CB reorder | 51.0 | 156 | 19.0 | 18.0 |

이는 샘플당 평균이며 request-byte 비율은 아니다. CB q24는 평균 10 scan batch 중 7.5 batch의 적재가 거부되고 끝에서 cache residency가 약 0.43GiB로 줄어든다. nominal 6GiB가 전 구간에서 실제 유지된다는 가정은 성립하지 않는다.

현재 cuCascade의 `get_available_memory()`는 capacity에서 예약을 포함한 회계량을 뺀다. active reservation 안에서 deallocate하면 `upstream_reclaimed_bytes=0`인 경로가 있어 페이지 해제와 표시 free 증가가 바로 대응하지 않을 수 있다(`cucascade/src/memory/reservation_aware_resource_adaptor.cpp:252,463`). 기존 실험의 정확한 원인으로 확정하려면 동시 계측이 필요하다.

다음 확인: eviction 직전·직후·관련 task 종료 시점에 실제 cache live bytes, reserved bytes, allocation 회계량, 해제된 page의 남은 참조를 함께 기록한다. reservation 유지, page shared ownership, 비동기 해제 중 무엇이 free 표시를 늦추는지 구분한다. min-free를 낮춰 숫자를 개선하는 것으로 시작하지 않는다.

**4. 컬럼 overlap이 실제 page 재사용으로 이어지는 비율을 본다.** 누적 page-census를 execution 마지막 snapshot끼리 차분했다. SET마다 찍힌 누적 값들을 합산하지 않았다.

| 벤치마크/조건 | 적재 | 서빙 | 축출 | 서빙/적재 |
|---|---:|---:|---:|---:|
| TH paging | 101.78GiB | 16.11GiB | 101.75GiB | 15.8% |
| TH reorder | 97.78GiB | 27.02GiB | 97.78GiB | 27.6% |
| CB paging | 34.40GiB | 28.91GiB | 34.40GiB | 84.0% |
| CB reorder | 32.23GiB | 33.14GiB | 32.23GiB | 102.8% |

이 값은 execution당 GPU 페이지 allocation 트래픽이며 padding을 포함한다. served는 전체·부분 서빙을 모두 세지만, 요청 hit 비율·유효 payload·물리 디스크 읽기 절약량은 아니다. 같은 페이지를 여러 번 쓰므로 100%를 넘을 수 있다.

TH의 l_shipmode/l_shipinstruct는 reorder에서도 각 4.56GiB를 적재하고 served는 0이다. l_quantity는 paging에서 13.08GiB 적재/2.14GiB served다. CB UserID는 0.90GiB 적재/8.90GiB served로 재사용 효율이 높다. 다만 TH 4GiB/CB 6GiB라는 예산 차이도 있으므로 모든 차이를 workload 특성에 귀속할 수는 없다.

다음에는 각 page request를 최초 미스, 예산에 따른 축출, 메모리 압력에 따른 축출, 적재 거부, filter/layout 불일치, 부분 서빙에서 사용되지 못한 조각으로 분류하고 후속 읽기 시간과 대응시켜야 한다. 현재 FA 로그에는 page ID와 lookup 결과 trace가 없어 별도의 untimed trace가 필요하다. 현재 partial-piece 탐색에는 처음 남아 있는 페이지가 index 64보다 뒤에 있을 때 탐색을 멈추는 경로도 있어, 큰 STRING row group의 tail만 남는 경우를 따로 검증해야 한다.

**5. cache ON에서 바뀌는 필터 위치와 중간 데이터량을 분리한다.** cacheable scan은 unfiltered column을 보관하기 위해 reader의 row filter를 post-decode 쪽으로 옮긴다. row group pruning과는 별도 결정이다.

| 대상 | baseline materialized rows | paging/reorder materialized rows | 최종 scan 출력 |
|---|---:|---:|---|
| CB q14/q25/q32 각 query | 13.172M | 99.997M | 조건 간 일치 |
| CB q37 | 0.672M | 20M | 조건 간 일치 |
| TH q12 | 119.318M | 375.006M | 조건 간 일치 |

중간 행 수 증가는 측정됐지만 디코딩 계산량이나 물리 I/O가 같은 비율로 증가했다는 의미는 아니다. 재사용하기 쉬워지는 이점과 중간 table·filter·reservation 비용을 함께 측정해야 한다. `compressed_bytes`는 완전 hit에서도 변하지 않는 planned demand이므로 실제 디스크 읽기 바이트로 사용하지 않는다.

다음 대조 조건은 cache OFF + 같은 reordered SQL 순서, 필요하면 cache OFF + 같은 post-filter 경로다. 앞 조건으로 SQL 순서 자체의 효과를, 뒤 조건으로 필터 위치 변경과 캐시 저장·재사용 효과를 구분한다. 새로운 reorder policy를 만들기 전에 기존 policy의 이득과 손실을 설명한다.

**시간 계측의 주의점.** read task 합은 TH 102.27→88.73→85.16s, CB 53.94→39.10→37.90s로 줄었다. populate는 TH 0→3.34→3.56s, CB 0→1.22→1.25s다. 이 값을 wall에서 빼서 손실을 계산하지 않는다. Computing은 HOST 대기·CUDA enqueue를 포함하므로 kernel만 가동된 시간이 아니다. reserve-slot share 적분도 실제 wall 손실이 아니라 예약 대기와 빈 worker가 겹치는 정도의 진단 지표다.

| 벤치마크/조건 | scan-active wall 구간 | other-only 구간 | 나머지 | reserve-slot share 진단 |
|---|---:|---:|---:|---:|
| TH baseline | 29.575s | 1.562s | 1.205s | 0.016s |
| TH paging | 28.182s | 1.534s | 1.140s | 0.955s |
| TH reorder | 26.891s | 1.564s | 1.198s | 0.847s |
| CB baseline | 25.437s | 3.266s | 2.135s | 2.788s |
| CB paging | 21.886s | 3.753s | 2.189s | 3.005s |
| CB reorder | 21.170s | 3.420s | 2.179s | 2.856s |

**재현 파일.** `analyze_headline_three_conditions.py`는 전체 query CSV와 `headline_reorder_contributions.png/pdf`, `analyze_headline_page_census.py`는 census summary/columns/queries/executions CSV, `analyze_headline_task_occupancy.py`는 task occupancy와 summary CSV를 생성한다. 모두 `pixi run -e duckdb-python python result/<script>.py`로 기존 로그만 분석한다.
