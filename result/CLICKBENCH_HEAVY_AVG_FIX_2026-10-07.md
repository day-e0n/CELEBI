# ClickBench q28/q29: 원인 확인 및 AVG 집계 개선

q28/q29의 비정상적으로 긴 실행 시간은 grouped AVG의 128비트 누산 경로가
주된 원인이었다. 같은 SQL과 같은 실험용 바이너리에서 누산 경로만 바꾼
비교로 확인했고, overflow를 방지하는 개선을 기본 실행 경로에 적용했다.

## 전체 query latency 비교

아래는 **캐시 OFF, q28→q29만 실행한 독립 진단**이다. 전체 ClickBench
43개 쿼리의 캐시 이력이나 CELEBI의 캐싱 효과를 나타내는 결과가 아니다.
동일 바이너리의 기존 누산 경로와 안전한 64비트 누산 경로를 비교했다.
각 조건 3회 실행 중 첫 실행을 제외한 2회 평균이다.

| Query | 기존 누산 경로 | 안전한 64비트 누산 | 시간 감소율 | Speedup |
|---|---:|---:|---:|---:|
| q28 | 17.0200 s | 1.2893 s | 92.42% | 13.20× |
| q29 | 40.3380 s | 11.9743 s | 70.32% | 3.37× |

시간은 `con.execute(sql).fetchall()`의 경과 시간으로, scan 합산 시간이나
여러 병렬 task의 operator work 합계가 아니다. 초기 비교는 타이머 안에서
짧은 SQL 파일도 기록했으며 양쪽에 동일하게 적용했다. 최종 검증에서는
이 파일 기록을 타이머 밖으로 옮겼다.

조건별 독립 프로세스는 하나이고 측정 표본은 두 개이다. 특히 q29의 최종
기본 경로 검증은 10.20 s와 15.33 s로 편차가 있으므로, 소수점 수준의
성능 차이나 신뢰구간을 주장할 수 없다. 큰 개선 폭은 동기화 진단,
진단 OFF 반복 측정, 최종 기본 경로 검증에서 모두 확인했다.

## 코드에서 발생한 일

두 쿼리의 공통 표현식은 `AVG(STRLEN(URL))`와 `AVG(STRLEN(Referer))`다.
`STRLEN`의 cuDF 결과는 INT32이고, 표현식 후처리가 SQL 반환형 BIGINT에
맞춰 INT64로 변환한다. grouped AVG는 SUM과 COUNT_VALID로 분해된다.
기존 `local_grouped_aggregate`는 AVG의 SUM 입력이 INT64/UINT64이면
입력 컬럼 전체를 DECIMAL128(scale 0)로 변환한 뒤 cuDF groupby에 전달했다.
이는 UserID 같은 큰 정수의 합계 overflow를 막기 위한 경로였지만,
문자열 길이처럼 합계가 충분히 작은 정수에도 적용됐다.

원인은 변환 비용만이 아니라 **DECIMAL128 입력을 받는 집계 경로의 비용**이다.
단일 worker에서 GPU stream을 각 API 단계 후 동기화한 진단 결과:

| Query | 기존 groupby.aggregate 단계 합계 | 개선 후 단계 합계 |
|---|---:|---:|
| q28 | 16.1773 s | 0.1215 s |
| q29 | 28.9961 s | 0.1774 s |

q29의 HLL 및 dictionary encoding을 포함한 key 준비는 개선 경로에서
합계 0.0880 s였다. 이를 주요 병목으로 설명할 근거는 없다. 정확한 CUDA
kernel/알고리즘 선택을 프로파일한 결과는 아니므로 atomic contention이나
특정 hash/sort 내부 구현을 확정 원인으로 주장하지 않는다.

## 적용한 수정과 안전성

`src/op/aggregate/gpu_aggregate_impl.cpp`에서 AVG용 64비트 입력의
최소값/최대값을 GPU reduction으로 구한다. N은 입력 행 수다.

- INT64: `N * min >= INT64_MIN` 및 `N * max <= INT64_MAX`를 확인한다.
- UINT64: `N * max <= UINT64_MAX`를 확인한다.
- 검사 계산은 host의 `__int128`으로 수행해 검사 자체의 overflow를 막는다.

이 보수적 조건은 모든 그룹과 부분 누산의 범위를 보장한다. 조건이
성립하면 원래 INT64/UINT64 입력으로 SUM을 구한다. 성립하지 않으면 기존
DECIMAL128 입력 경로를 유지한다. 모두 NULL인 입력도 NULL 합계와
0 count를 유지한다.

빠른 경로에서도 **집계 후의 작은 partial SUM 결과를 DECIMAL128로 변환**한다.
따라서 batch마다 누산 폭이 달라도 merge 입력 형식은 동일하다.
일반 SUM과 AVG용 SUM도 독립 요청을 유지한다. q28/q29 SQL은 그대로다.

개선은 기본 ON이며 `SIRIUS_GROUPBY_SAFE_AVG_INT64=0`으로 기존 경로와
대조할 수 있다. `SIRIUS_GROUPBY_DIAGNOSTIC=1`은 단계별 동기화를 추가하므로
원인 분석용으로만 사용하고 latency 측정에서는 켜지 않는다.
실험용 sort 강제/키 dictionary 비활성화 경로는 최종 코드에서 제거했다.

## 최종 빌드 및 캐시 ON 검증

정리한 최종 바이너리로 별도 프로세스에서 다시 3회씩 실행했다.
query 순서는 두 조건 모두 [28,29]이고 캐시 ON 조건의 future hint horizon은
각 execution이다. 전체 43-query reorder workload와 다른 진단 조건이다.

| 최종 빌드 조건 | q28 평균 | q29 평균 |
|---|---:|---:|
| 기본 설정, cache OFF | 1.2387 s | 12.7667 s |
| page cache ON + future-aware eviction | 1.8450 s | 13.1816 s |

이 작은 workload에서 캐시 ON이 더 빠르다고 결론 내릴 수 없다. q29의
캐시 ON 실행에는 projection에서 약 4.04 GB 할당을 요청하다 OOM으로
task를 재실행한 기록이 한 번 남았다. AVG 개선은 이 문제까지 해결하지 않는다.

검증:

- 전체 99,997,497행을 읽어 CPU DuckDB q28/q29 정답을 새로 계산했다.
- 동기화 진단, 반복 측정, 최종 기본 경로 및 future eviction 실행의
  **28개 결과 세트가 CPU와 일치**했다. 정수/문자열은 정확히 비교하고,
  float은 relative 1e-8 / absolute 1e-6 tolerance를 적용했다.
- signed/unsigned 64비트의 안전한 값, 최댓값/최솟값, 양·음수 혼합,
  모두 NULL인 값의 partial SUM 형식과 정확성을 검증했다.
- grouped aggregate C++ 테스트: **23 cases, 168 assertions 모두 통과**.
- 수정 파일의 pre-commit 검사와 최종 빌드 완료.

## q29에 남은 병목과 다음 조사 대상

기존 단일 worker operator 진단에서 scan work는 q28 1.5167 s,
q29 1.2564 s였고, q29 projection은 9.6129 s였다. AVG 집계를 개선한 뒤
남는 q29의 큰 비용은 정규식 기반 도메인 추출을 포함한 projection이다.
이는 현재 남은 시간을 정확히 모두 분해한 kernel 측정은 아니다.

실행 중인 경로는 `cudf::strings::replace_with_backrefs`이며
`enable_regex_jit_impl=false`이다. 기존 선택적 JIT 구현의 코드에는:

1. evaluator의 stream/memory resource를 전달하지 않고
   `transform_extended` 기본값을 사용하는 문제가 있다.
2. optional `www.`의 정규식 backtracking을 재현하지 않는다.
   `http://www./x`에서 정규식 결과는 `www.`이지만 현재 구현은 www.를
   제거한 뒤 빈 hostname으로 판단해 원문을 반환한다. CPU 정규식의
   해당 결과는 별도 reference artifact로 확인했다.
3. UTF-8에서 반복되는 `length()/operator[]/substr()`의 비용과 byte 경계를
   검토해야 한다. NULL, Unicode, 비매칭 URL, newline의 의미도 보존해야 한다.

현재 꺼져 있는 JIT 경로의 문제는 실제 측정에 사용한 정규식 경로의
오류 원인을 뜻하지 않는다. 이를 수정하고 충분히 검증하기 전까지 켜지 않았다.

그 다음은 projection의 일시 메모리 예약과 동시 실행량이다. q29는
`MIN(Referer)`도 구하므로 긴 원문 문자열을 실제로 보유해야 한다.
Parquet coalescer가 row group을 통째로 내보내는 경로에서는 목표 batch
byte 수만 낮춰도 큰 개별 row group이 자동으로 나뉘지 않는다.
이 부분은 별도 최적화 대상으로 남아 있다.

## 이전 그래프와 논문에서의 해석

기존 FULL_CACHE_PIN 전체 43-query workload에서 q28 baseline/CELEBI는
16.7314/16.6773 s, q29는 40.3707/40.5019 s였다. q29의 별도 byte hit trace는
CELEBI 12.5430%였다. cache hit가 있어도 128비트 평균 집계와 정규식 비용은
남았으므로 hit ratio만으로 query latency를 설명할 수 없었다.

4 workers의 기존 q28 scan work 28.1739 s, q29 scan work 49.4659 s는
병렬 task의 Computing interval 합계다. overlap과 GPU 대기를 포함하므로
query의 처음부터 끝까지 경과 시간과 같지 않다.

이번 AVG 개선은 Baseline과 CELEBI에 공통으로 적용되는 엔진 개선이다.
**92.4%/70.3%를 CELEBI 캐싱만의 이득으로 발표하면 안 된다.** 개선 후
논문용 전체 workload 비교는 양쪽을 동일한 새 바이너리로 다시 측정해야 한다.
기존 그래프나 누락 query 값을 새 진단 결과로 임의 교체하지 않았다.

## 환경과 재현 파일

- Physical GPU 1: RTX A5000 24 GiB, 1 visible GPU.
- Driver 595.71.05; CUDA runtime 13.2.75; libcudf 26.06.01, RMM 26.06.00.
- GPU usage limit `20GB`, host capacity `32GB`, 4 GPU pipeline workers.
- cache ON 예산 6 GiB, page 단위 16 MiB, min-free 6 GiB.
- late materialization ON, O_DIRECT ON, prefetch OFF.
- CPU fallback OFF, regex JIT OFF, hit/page trace OFF인 latency 측정.
- 입력: `/mnt/nvme/clickbench/hits_v2.parquet`, 99,997,497 rows, 10 row groups.
- 기존 별도 ClickBench 실험 종료를 기다린 뒤 측정했다. 최종 검증은
  06:19:15 UTC에 종료했고 이후 발견한 GPU 0 실험은 06:19:39에 시작했다.

Artifacts under `experiment/CLICKBENCH_HEAVY_DEEP_20261007/`:

- `compare/{phase_auto,phase_native,latency_auto,latency_native}`: 원인 대조.
- `compare_final/{latency_default,latency_future}`: 최종 빌드 검증.
- 각 case의 `metadata.json`, `queries.csv`, `results.json`, `q28.sql`, `q29.sql`.
- `cpu_reference.json`, `analysis.json`, `regex_semantic_reference.json`.
- `unit_groupby_final.log`, `lint_analysis.log`, `build_final_v2.log`.
- `native_trial/sirius.duckdb_extension`: 동일 바이너리 대조 snapshot.
- `final/sirius.duckdb_extension`: 최종 검증 snapshot.

최종 extension SHA256:
`d64c9264269a0a35b92a156d5fba2ef64cf818952a84ba9a4809a46d01e9e882`.

Reproduction runners:
`scripts/run_heavy_avg_comparison.py`,
`scripts/diagnose_clickbench_heavy_concurrency.py`,
`scripts/analyze_heavy_avg_comparison.py`.

초기 narrow-length SQL 시도 두 건은 extension filename/GPU 점유로
쿼리 실행 전에 실패했다. 이전 Nsight Systems 시도는 CUDA runtime의
호환 문제(`Unknown runtime API function index: 509`)로 kernel 데이터를
수집하지 못했다. 이 자료들은 성능 평균과 원인 결론에서 제외했다.
