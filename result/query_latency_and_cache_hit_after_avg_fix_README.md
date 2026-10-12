# ClickBench 재측정 및 기존 TPC-H 결과를 합친 그림

ClickBench 43개 전체를 AVG 개선 빌드로 다시 측정했다. q24·q28·q29 포함.
TPC-H SF50 22개는 사용자 요청에 따라 기존 FULL_CACHE_PIN 측정 결과를 유지했다.

| ClickBench | Baseline | Page caching | CELEBI |
|---|---:|---:|---:|
| 전체 query latency 합계 | 43.1714 s | 39.3445 s | 37.8387 s |
| 일반 cache hit ratio, 바이트 기준 | — | 22.5286% | 37.0061% |

CELEBI의 전체 시간 감소율은 baseline 대비 12.35%다.
CELEBI는 page caching + small-first-big-last reorder + future-aware eviction이다.

시간은 execute().fetchall() 경과 시간으로, 모든 operator와 재시도 비용을 포함한다.
조건별 독립 프로세스 한 개에서 3회 실행하고 첫 실행은 제외했다.
쿼리별 2·3회차 평균을 query-number 순서로 누적했으며 실제 reorder 실행 순서의
시간 축은 아니다. trace OFF 실행에서 latency를, 동일 설정의 별도 trace ON
실행에서 cache hit ratio를 얻었다. 히트율은 전체 요청 decoded value bytes 중
캐시가 공급한 바이트 비율이다. 반복 요청을 포함하며 STRING offsets, NULL masks,
allocation padding은 제외한다. 조건부 캐시 경로 성공률 100% 지표가 아니다.

패널 내부 크기는 세 개 모두 5.3 × 3.5 inch이며 latency 두 패널의 y축 범위도 동일하다.

## 실험 조건

- 물리 GPU 1, RTX A5000 24 GiB, GPU pipeline workers 기본 4개.
- GPU usage limit 20GB, host capacity 32GB, ClickBench 공유 cache budget 6 GiB.
- page 단위 16 MiB, min-free 6 GiB, O_DIRECT ON, prefetch OFF, late materialization ON.
- 입력 /mnt/nvme/clickbench/hits_v2.parquet; 원본 SQL과 query 목록 유지.
- 새 extension SHA256: d64c9264269a0a35b92a156d5fba2ef64cf818952a84ba9a4809a46d01e9e882.
- 기존 TPC-H extension SHA256: 0b53e79926ac5457625b95c146aef350ac5b4d74978b302e3278303098e79a19.
- 측정 간 다른 GPU 작업의 overlap은 감지되지 않았다.

## 캐시 추적의 종료 오류와 처리

초기 trace ON의 paging 및 future 조건은 43개 쿼리 3회를 모두 마친 뒤
DuckDB 연결 종료 중 SIGSEGV가 발생했다. 스택은 scan manager의 drop_all_pages에서
RMM cuMemFreeAsync를 호출하는 경로를 가리킨다. 이 두 실행을 그림에서 제외했다.

Paging은 같은 설정의 별도 재측정이 정상 종료해 그 trace를 사용했다.
Future의 일반 종료 재측정에서도 같은 오류가 반복됐다. 최종 future trace는
같은 엔진 바이너리·쿼리·순서·예산으로 다시 실행하되, 129개 query 결과와
worker_finished journal이 저장됐는지 확인한 뒤 CUDADeviceSynchronize와
로그 flush를 수행하고 격리 worker 프로세스를 종료했다. 기존 엔진의 마지막
destructor만 우회하는 측정 후 종료 처리이며 쿼리 사이의 캐시 상태를 바꾸지 않는다.
엔진의 일반 종료 버그가 해결됐다고 주장하지 않는다.

최종 trace에서 q1~q43 × executions 2·3의 로그 완전성과 바이트 카운터 범위를
검사했다. 사용한 두 trace case의 비교 설정을 latency case와 대조했다.
미완료 실행을 완결된 것으로 수정하지 않았으며 모든 원본 실패 기록을 보존했다.
실행 경로와 종료 정책은 provenance JSON에 명시했다.

## 파일

- query_latency_and_cache_hit_after_avg_fix.png / .svg / .pdf
- query_latency_and_cache_hit_after_avg_fix_query_latency_means.csv / _query_latency_samples.csv
- query_latency_and_cache_hit_after_avg_fix_cache_hit_summary.csv / _cache_hit_per_query.csv
- query_latency_and_cache_hit_after_avg_fix_provenance.json
- 원본 재측정: /home/dy1013/sirius_multi-gpu/experiment/AVG_FIX_CB_20261007
