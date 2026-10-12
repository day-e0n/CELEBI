**비교 목적.** 기존 `pin_table(tier='gpu')`는 필요한 컬럼을 사전 디코딩해 GPU에 보관하므로 페이지 캐시 성능을 설명하는 대조군으로 적절하다. 먼저 같은 입력이 모두 메모리에 들어가는 쿼리로 캐시 공급 경로의 비용을 비교하고, 별도로 같은 저장 예산에서 workload 정책을 비교한다. 새 GPU 실험이나 엔진 변경은 이 검토에서 수행하지 않았다.

**100%의 범위.** 전체 요청 100건 중 캐시 경로로 간 20건이 모두 hit이면 전체 hit는 20%다. 캐시 대상 요청 내부의 100%는 조건부 비율이며 workload 전체 비율로 발표하지 않는다. 전체 decoded-byte 공급률은 동일한 scan 단계에서 `캐시가 공급한 입력 바이트 / 모든 경로가 공급한 입력 바이트`로 계산한다. miss·partial과 별도 읽기 경로를 포함하고, 쿼리별 %의 단순 평균 대신 바이트 합으로 가중한다. 저장했던 데이터 중 다시 사용한 비율은 별도 지표다.

현재 HIT_*의 `[cache-hit]`는 metadata reader 경로의 조립 완료 지점에서 `total_bytes`, `read_bytes`를 기록한다. latest A_*의 late-materialization `read_row_numbers()`는 그보다 앞에서 반환한다. 수동 pin의 cached-provider 경로도 이 로그를 우회한다. 따라서 기존 집계만으로 최신 workload 전체 또는 pin의 공급률을 계산하면 안 된다.

근거: [일반 경로 로그](/home/dy1013/sirius_multi-gpu/src/op/scan/parquet_gpu_ingestible.cpp:2310), [late-refetch early return](/home/dy1013/sirius_multi-gpu/src/op/scan/parquet_gpu_ingestible.cpp:1760), [cached-provider 별도 경로](/home/dy1013/sirius_multi-gpu/src/op/scan/gpu_ingestible.cpp:29).

| 실험 | 조건 | 확인할 내용 |
|---|---|---|
| 전체 hit 상태의 공급 경로 비교 | baseline / 페이지 캐시 warm / GPU pin, 같은 쿼리·데이터·GPU 한도·planner 설정 | pin에서도 느리면 읽기를 생략한 뒤 남는 비용 확인. page만 느리면 layout·copy/concat·filter·batch scheduling 차이 분석 |
| 제한된 저장 예산의 workload 비교 | TH 4GiB / CB 6GiB 이하의 static pinned column subset vs dynamic pages | 같은 저장 용량에서 어떤 방식이 요청을 더 많이 공급하고 전체 시간을 줄이는지 |

**첫 비교 대상.** ClickBench q4(UserID AVG), q14(SearchPhrase GROUP BY + UserID COUNT DISTINCT), q26(SearchPhrase ORDER BY). q4는 fixed-width, q14·q26은 문자열과 집계/정렬을 포함한다. 과거 HIT에서 q14·q26의 full hit가 관찰됐지만 새 코드에서도 실제 전체 공급 여부를 검증한다. 쿼리가 요청하는 전체 컬럼 및 행 범위가 예산에 들어가는 경우만 전체 hit 대조군으로 채택한다. q24는 별도 row-id refetch·OOM·cache drop이 있어 이후 따로 분석한다.

현재 late materialization scan은 `FILE_ROW_NUMBER`/`FILE_INDEX` 가상 컬럼을 요청한다. 수동 pin은 parquet schema의 실제 컬럼만 보관하지만 pin matching은 요청에 포함된 가상 컬럼까지 superset을 요구하므로 해당 scan의 pin 매칭이 실패한다. 전체 실제 컬럼을 pin했다고 q24가 pin에서 공급되는 것은 아니다. 이 결론은 소스 경로 검토이며 새로운 pin 실행으로 검증한 결과는 아니다. 초기 대조군은 가상 row-id를 요구하지 않는 쿼리를 선택한다. 별도 진단에서 LM을 OFF로 할 경우 모든 조건에 동일하게 적용하고 current LM ON headline과 별도 표시한다. dynamic-filter reuse gate도 pin에 적용된다.

근거: [실제 컬럼으로 pin 생성](/home/dy1013/sirius_multi-gpu/src/sirius_extension.cpp:867), [가상 컬럼 처리](/home/dy1013/sirius_multi-gpu/src/op/scan/scan_plan.cpp:263), [요청 컬럼 전체 매칭](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:3886).

**수동 pin의 성격.** `cols=[...]`는 선택 컬럼의 전체 행을 pin한다. 페이지 단위 임의 row subset을 pin하는 API는 없다. 동일 parquet file set이고 요청 컬럼 전체가 pin의 subset이어야 cached provider를 배정한다. 한 컬럼이 빠지면 scan 전체가 reader로 돌아갈 수 있다. 이 차이는 static column pin과 dynamic page cache의 정책 차이이며, 단순 lookup 비용으로 해석하지 않는다.

수동 GPU pin은 원본 decoded 컬럼 chunk를 move하여 저장한다. provider의 view 생성부터 scan output까지 완전히 zero-copy인 것은 아니다. generic owner release는 새 table을 복사하며 필터가 있으면 gather 등도 수행한다. 따라서 page와 pin의 시간 차이를 lookup/populate 오버헤드만으로 단정하지 않는다.

근거: [pin API와 materialize](/home/dy1013/sirius_multi-gpu/src/sirius_extension.cpp:1128), [원본 chunk 저장](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:4571), [cached provider](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:173), [generic owner copy](/home/dy1013/sirius_multi-gpu/src/include/op/scan/owning_table_view.hpp:193), [매칭 조건](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:6419).

**용량 통제.** pin chunk의 GPU allocation도 cache resident 집계에는 포함된다. 그러나 일반 budget sweep은 수동 chunk 자체를 축출하지 못하므로 4/6GiB 설정만으로 pin을 한도 이하로 만들 수 없다. pin할 컬럼을 사전에 선택하고 실제 resident allocation이 한도 이하인지 확인한다. 전체 GPU pin이 예산을 초과하면 별도 대조 실험으로 표기하거나 실행 불가로 보고한다. OOM emergency drop은 수동 GPU pin chunk도 제거하므로 계속 상주했다고 가정하지 않는다.

근거: [pin allocation 집계](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:3257), [budget 및 초과 경고](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:5482), [OOM cache drop](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:5839).

기존 column-size metadata의 합은 TH SF50 약 52.215GiB(lineitem 약 35.983GiB), ClickBench 105컬럼 약 66.609GiB다. 문자열은 표본 평균 길이 추정이고 mask/padding 등을 포함한 최신 실측 allocation이 아니다. 그러나 전체 컬럼 GPU pin을 현재 4/6GiB 예산에서 실행하는 조건은 성립하지 않는다. [TH 추정 자료](/home/dy1013/sirius_multi-gpu/experiment/column_bytes/tpch_sf50.json:12), [CB 추정 자료](/home/dy1013/sirius_multi-gpu/experiment/column_bytes/clickbench.json:2).

**측정 절차.** 같은 commit/build, physical GPU, GPU 20GiB 한도, HOST 한도, 데이터 file set, SQL, late materialization, 예약 추정, prefetch OFF, O_DIRECT, query order와 skip list를 고정한다. 페이지 warm과 pin setup을 각각 별도 측정하고 query-only latency와 `setup + N회 실행`을 함께 제시한다. 초기 실행과 반복 실행을 섞어 평균하지 않는다. pin 대조군에서는 자동 page populate를 끄고 기존 chunk-backed provider를 사용하도록 provider 설정을 명시한다. trace 계측 실행과 최종 latency 실행은 분리한다. correctness와 실제 provider 배정/reader 호출 여부를 확인한다.

같은 SQL이라도 pin의 chunk provider와 page/reader provider에서 읽는 row groups와 batch 수가 달라질 수 있다. 비교 전 실제 input rows/RGs, 요청 컬럼, GPU placement, batch 크기를 확인하고 차이가 있으면 함께 보고한다. full-hit 대조의 시간 차이는 populate 비용을 포함하지 않으며, populate 비용은 초기 적재 실행에서 별도 측정한다.

workload 비교는 동일 arrival order를 먼저 사용한다. reorder 및 미래 인지 축출은 별도 조건으로 추가한다. 쿼리마다 필요한 모든 컬럼을 미리 pin하고 unpin하는 방식은 미래 요청을 알고 준비하는 대조군이므로, bounded dynamic page cache의 정책 경쟁자로 취급하지 않는다.

**함께 보여줄 값.** query-only 및 setup 포함 wall time, 공통 scan 단계의 decoded-byte cache 공급률, 실제 cache resident 및 전체 GPU peak allocation, reader의 압축 I/O bytes, cache copy/concat·post-filter·주요 연산 시간, OOM/retry/drop 횟수. 겹치는 task 시간 합은 wall에서 직접 빼지 않는다. 일반 `[cache-hit]`에 pin/late-refetch가 빠지므로 공통 scan/provider 단계 카운터를 보완해야 한다.

**기존 runner의 함정.** `performance_test.py --pin gpu|host`는 지원하지만 `--pin-columns`는 없다. grouped mode는 쿼리별 pin/unpin, sequential mode는 TPC-H 전체 query mapping의 union을 pin한다. pin setup은 query timer 밖이고 pin time 별도 CSV 기록은 없다. pin column mapping과 view 등록이 TPC-H에 묶여 있으므로 ClickBench에 `--query-set clickbench_queries --pin gpu`만 붙이는 것으로 올바른 비교가 되지 않는다. 기존 CB runner와 동일 source 등록/설정을 유지하며 명시적인 `pin_table(..., cols=[...])` 조건을 추가하는 준비가 필요하다.

근거: [grouped pin](/home/dy1013/sirius_multi-gpu/test/tpch_performance/performance_test.py:413), [sequential union pin](/home/dy1013/sirius_multi-gpu/test/tpch_performance/performance_test.py:456), [query-only timer](/home/dy1013/sirius_multi-gpu/test/tpch_performance/performance_test.py:341), [TPC-H column mapping](/home/dy1013/sirius_multi-gpu/test/tpch_performance/tpch_pin_columns.py:31).

HOST pin도 decoded 데이터를 보관하며 query에서 H2D 전송이 필요하다. 필요하면 별도 대조군으로 쓸 수 있으나, 이번 첫 비교는 GPU pin으로 시작한다. HOST pin은 사용자가 제거한 evicted decoded page의 HOST secondary cache와 다른 기존 기능이다.
