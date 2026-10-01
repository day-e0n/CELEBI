# 페이지 캐시 작업 인수인계

ClickBench / TPC-H SF50 에서 GPU 페이지 캐시(CELEBI)를 성능 개선 장치로 만드는 작업.

---

## 0. 사용자가 요구한 것 (원본 요구사항)

### 재정렬 기법 — 사용자가 직접 설계

| | 요구 | 상태 |
|---|---|---|
| A1 | 인접 쿼리가 같은 컬럼을 읽게 붙여 캐시 재사용 (overlap) | 구현됨, 모든 정책의 기반 |
| A2 | 작은 컬럼끼리는 겹치되 서로 다른 큰 컬럼은 멀리 | `big-diff-apart` → +15.5% 손해 |
| A3 | 같은 큰 컬럼을 읽는 쿼리끼리 붙이기 | `big-same-adjacent` → -1.3% (오차 범위) |
| A4 | 등급1 = 큰 컬럼 안 읽음(겹침 순) / 등급0 = 큰 컬럼 읽음(뒤로) | `small-first-big-last` → natural -15.4%, **5종 평균 +7.2%** |
| A5 | 동점: ① fixed overlap byte 큰 것 ② 동점이면 variable overlap 적은 것 ③ fixed 0이면 variable overlap 순 | 구현됨 (`_resident_pair_overlap_rank`) |
| A6 | 큰 컬럼은 겹침 계산에서 제외 | 구현됨 (`_resident_signature`) |
| A7 | **overlap 아이디어를 바꾸지 말 것. 새 기법을 만들지 말 것** | 지켜짐 |
| A8 | 큰 컬럼 기준 = 컬럼 전체 크기 | `SIRIUS_REORDER_BIG_COLUMN_RATIO` (기본 1.0 = 예산) |

### 캐시 구조

| | 요구 | 상태 |
|---|---|---|
| B1 | 16MB 페이지 단위 | 구현됨 |
| B2 | 페이지를 offset 기준으로 자를 것 (평균 금지) | 구현됨. 평균 방식은 0.09~80.98MiB 페이지를 만들었음 |
| B3 | 16MB 안 되면 더미로 채워 16MB 만들기 | **미구현** |
| B4 | row group 의 페이지가 전부 없어도 쓸 수 있게 | **미구현.** "전부 아니면 포기"가 남아 있다. cuDF 가 `set_row_groups`+`skip_rows` 조합을 금지해 부분 읽기가 막힌다고만 확인했고, 우회로는 안 찾아봤다 |
| B5 | fixed·variable 둘 다, 경계 없이 하나의 예산 | **완료** (커밋 `c47a67d5`) |
| B6 | 너무 큰 컬럼은 아예 안 올리기 | 시도 → `SKIP_OVERSIZED=1` 로 203초(캐시 무용지물). 기각 |
| B7 | variable 먼저 축출 | 구현됨 (2패스) |

### 실험 조건

| | 요구 | 상태 |
|---|---|---|
| C1 | baseline / 캐시ON·재정렬없음 / 캐시ON+재정렬 — **이 셋만** | 완료 |
| C2 | ClickBench + TPC-H SF50 둘 다 | 완료 |
| C3 | natural 순서를 기준으로 (worst 는 버림) | 완료 |
| C4 | 다른 재정렬 기법은 절대 하지 말 것 | 지켜짐 |
| C5 | 모든 쿼리가 다 돌아야 함 | 완료 — 연산 4종 추가, 스킵 0 |
| C6 | 정확성 검사 | 완료 — TPC-H 일치, ClickBench 9개는 ORDER BY 동점 |

**B3·B4 가 남은 미구현 요구사항이다.** 특히 B4 는 지금 URL 페이지가 적재 568장 중
483장 축출되고도 서빙 0 인 이유와 직결된다.

---

## 1. 이번 세션에서 추가로 생긴 요구사항

### R1. 캐시는 설정한 예산만큼만 쓴다 — **완료** (커밋 `c47a67d5`)

6GB 로 설정하면 캐시가 쥐는 모든 디바이스 메모리의 합이 6GB 이하여야 한다.
수정 전 실측 9.52GB. 원인은 예산이 `_pages` 하나만 보고 있었고, 나머지 두 저장소
(`data_batches_by_column`, `variable_width_pages_by_column`)는 아무도 안 셌기 때문.

고친 지점 일곱 군데:

1. 축출이 `_pages_bytes` 만 예산과 비교 → 캐시 전체 합(공유 버퍼는 포인터로 중복 제거)
2. `merge_columns_into_chunk` 가 `index_variable_width_columns_for_chunk` 를 안 불러서
   STRING 컬럼이 페이징 없이 통째 청크로 보관 (3.03GB)
3. `variable_width_page_cache_budget_bytes_per_gpu()` 가 "남은 몫" 대신 전체 예산 반환
   → 6GB 설정이 6GB 캡 두 개로 동작
4. 그 차감에서 `_pages` 몫 누락
5. variable 페이지가 어떤 축출에도 없었음 → 같은 자리에서 **엔트리 단위로** 축출
   (페이지 단위는 과거에 캐시 히트 45% 손실 전례. 단위를 바꾸지 말 것)
6. 축출이 페이지 삽입 때만 호출 → 캐시가 자라는 모든 경로(6곳)에서 호출
7. `merge_columns_into_chunk` 가 엔트리를 키운 뒤 바로 `return` 해서 모든 축출 지점을
   건너뜀. ClickBench 는 43개 쿼리가 한 엔트리를 공유하므로 **이 경로가 주력**

검증: 494 샘플 전부 6.00GB 이하, 초과 0건.

### R2. OOM 이 나지 않는다 — **완료** (코드 수정 불필요)

`--min-free-bytes-per-gpu 6GB` 로 OOM 4회 → 0회. 기본값이 `0`(꺼짐)이라 아무도
안 쓰고 있었다. 러너 인자이고 환경변수는 `SIRIUS_FIXED_PAGE_CACHE_MIN_FREE_BYTES_PER_GPU`.

trade-off: min_free 를 키우면 축출이 잦아져 scan 절약이 줄어든다.
- min_free 2GB: scan read 59.90s (최선), OOM 4회, 벽시계 19.15s
- min_free 6GB: scan read 68.54s, OOM 0회, 벽시계 19.00s

### R3. 성능이 확실하게 개선된다 — **미달성**

현재 최선 19.00s, baseline 18.63s. **+2.0% 손해.**
어떤 예산(1~14GB), 어떤 min_free 로도 baseline 을 못 넘는다.

### R4. 재정렬 — overlap 아이디어 유지, 새 기법 만들지 말 것

`small-first-big-last` 가 사용자 설계. 도착 순서 5종 평균 **+7.2%** 로 이득 아님.
TPC-H 는 기본 설정에서 순서를 아예 안 바꾼다(예산 초과 컬럼 없음).
임계값은 `SIRIUS_REORDER_BIG_COLUMN_RATIO`(기본 1.0).

**R3 이 해결되기 전에는 재정렬을 더 만지지 말 것.** 키울 이득이 음수다.

---

## 2. 원인 — 측정으로 확정

캐시는 I/O 를 실제로 줄인다. 41개 쿼리 기준 parquet read 73.13 → 68.09초 (−5.04s).
43개 전체(q28·q29 포함)면 73.09 → 61.78초 (−15.5%).

그 이득이 세 곳으로 빠져나간다:

| 빠져나가는 곳 | 크기 (3실행 합) |
|---|---|
| **캐시 적재 (populate)** | **+3.26s** (그중 2.52s 는 콜드 실행) |
| Reserving — 메모리 예약 대기, **동시성 1.00배 = 완전 직렬** | +0.83s |
| Queued | +0.44s |

캐시 적재는 `auto_cache_materialized_table()` 이고 **`[scan-audit]` 로그 창 밖**에 있어
지금까지 어떤 계측에도 안 잡혔다. 16MiB 페이지로 잘라 GPU 에 복사하는 비용.

### 캐시 점유량은 원인이 아니다

캐시를 9.52 → 6.00GB 로 3.5GB 줄여도 GPU peak 18.62 → 18.57GB, 벽시계 19.19 →
19.15초로 변화 없음. "캐시가 메모리를 뺏어서 느리다"는 가설은 이 측정으로 반증됐다.

---

## 3. 측정 규칙 — 반드시 지킬 것

### 지표

**`total_ms`(GPU 연산자 시간 합)를 응답시간으로 읽지 말 것.** 동시에 도는 태스크들의
시간을 중복해서 더한다(ClickBench 2.85배). 캐시를 켜면 그 중복이 줄어 −37% 처럼
보이지만 벽시계는 −0.6% 다.

벽시계는 두 runner 가 `wall=...s` 로 쿼리마다 찍는다 (이 작업에서 추가).
겹침 제거가 필요하면 텔레메트리 span 의 합집합을 쓸 것.

### 실행

```bash
export BENCH_PREFETCH_CACHE=1        # 없으면 캐시가 꺼진 채 돈다. 전에 이걸로 한 번 틀렸다
ORD=$(seq 1 43 | grep -vxE "28|29" | paste -sd,)   # q28/q29 제외 41개
pixi run -e duckdb-python python scripts/run_clickbench_breakdown.py \
  --condition celebi_fixed_variable --executions 3 --order "$ORD" --no-reorder \
  --variable-cache-budget "" --min-free-bytes-per-gpu 6GB --output experiment/<name>
```

- `--skip-queries` 는 `--order` 에 덮인다. 순서 문자열에서 직접 빼야 한다
- 실행 1회차는 콜드, 2~3회차 평균을 쓴다
- 벤치마크는 한 번에 하나만 (NVMe 경합)
- 빌드는 `CMAKE_BUILD_PARALLEL_LEVEL=6` 로 제한 (공유 서버)

### 계측 (현재 소스에 들어 있음, 커밋 안 함)

- `[page-store] resident ... pages_bytes= chunk_bytes= varpage_bytes= total_bytes=`
  — 캐시 세 저장소의 실제 점유. 쿼리 끝마다
- `[scan-audit] ... stage_prep_us=` — read 이전 준비 구간
- `[scan-stage] materialize_us= postfilter_us= make_batch_us=` — scan 연산자 3단계
- `[scan-populate] populate_us=` — 캐시 적재 비용

---

## 4. 분석 스크립트

| 스크립트 | 용도 |
|---|---|
| `scripts/parse_quent_operator_breakdown.py` | 텔레메트리 → 연산자별 CSV |
| `scripts/breakdown_full_stack.py <run>` | 쿼리/태스크/연산자 3계층 |
| `scripts/breakdown_all_queries.py <run>` | 쿼리 43개 × 상태별 |
| `scripts/parse_page_trace.py <run>` | 페이지 적재/서빙/축출 (SIRIUS_PAGE_TRACE=1 필요) |
| `scripts/plot_query_gantt.py` | q1~q43 간트, 캐시 유무 비교 |
| `scripts/plot_executor_gantt.py [t0 t1]` | 파이프라인별 간트, 구간 확대 |

---

## 5. 다음에 할 일

1. **캐시 적재 3.26초를 줄인다.** 가장 큰 비용이고 아직 손 안 댔다.
   - 축출된 페이지가 재적재되는 사이클이 여기 들어 있다 (URL 페이지가 적재 0.25초 만에
     `pass=oversized` 로 축출되고 다시 적재되는 것을 페이지 추적으로 확인)
   - 서빙되지 않을 페이지를 애초에 안 들이는 판정이 필요
2. **Reserving 0.83초.** 동시성 1.00배라 전부 임계 경로. 캐시가 6GB 를 쥐는 한
   피하기 어렵다는 것까지 확인됨
3. 1·2 가 해결된 뒤에 R4(재정렬)를 다시 본다

---

## 6. 하지 말 것 (전례가 있음)

- 실험 설정(예산·조건·skip·min_free)을 말없이 바꾸지 말 것
- 측정 전에 원인을 단정하지 말 것. 이 작업에서 다섯 번 틀렸다
- 주석을 근거로 쓰지 말 것. `variable_width_page_cache_budget_bytes_per_gpu()` 의
  주석은 "하나로 합쳤다"고 하는데 코드는 안 합쳐져 있었다
- variable 페이지를 페이지 단위로 축출하지 말 것 (캐시 히트 45% 손실 전례)
- 용어: *page* 는 이 캐시의 16MiB 단위. OS 페이지 캐시도, parquet 페이지도 아니다.
  컬럼은 *fixed-width* / *variable-width*, 그냥 "fixed" 라고 쓰지 말 것
