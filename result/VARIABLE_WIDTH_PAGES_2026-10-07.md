# STRING 컬럼을 16 MiB GPU cache page로 나누는 방식

2026-10-07 코드 읽기 및 기존 로그 감사. 엔진·설정 변경과 GPU 실험은 하지 않았다.
이 문서의 page는 OS page나 Parquet encoded page가 아니라 Sirius decoded GPU cache page다.

## 현재 headline 실험에 해당하는 flat page store

STRING의 각 행 길이는 달라도 **GPU allocation 크기를 16 MiB로 고정**할 수 있다.
한 page에 들어가는 **행 수를 실제 문자열 길이에 맞춰 다르게** 정하면 된다.
같은 row group 안에서도 짧은 문자열 구간은 많은 행, 긴 문자열 구간은 적은 행을 담는다.
Fixed-width 컬럼과 STRING 컬럼의 page 경계가 서로 같을 필요도 없다.

현재 `insert_pages_from_view()`는 STRING을 평균 길이로 자르지 않는다.
cuDF STRING 컬럼의 offsets가 이미 누적 chars byte 위치를 담고 있어 이를 그대로 사용한다.
row group의 모든 offsets를 한 번 HOST로 복사하고, 각 page의 마지막 행을 binary search로 찾는다.
INT32와 INT64 offsets를 모두 처리한다.
같은 `(file, column, row_group, rows)`의 page 경계는 `_page_layouts`에 보관하여 다음 populate에서 재사용한다.

코드 근거:

- 기본 16 MiB: `src/scan_manager/sirius_scan_manager.cpp:1946`.
- STRING의 실제 offsets 복사와 layout 재사용: 같은 파일 `:5091–5140`.
- actual footprint에 대한 binary search: 같은 파일 `:5141–5171`.
- page는 row group별로 나뉘며 `[row_offset, row_offset + num_rows)`를 기록: 같은 파일 `:5050`, `:5357–5362`.

## Page 용량 계산과 저장되는 내용

행 구간 `[a, b)`에서 `N = b - a`, `C = offsets[b] - offsets[a]`, offset 원소 폭을 `W`라고 하자.
`A(x) = ceil(x / 256) * 256`, `M(N)`은 cuDF의 N행 null-mask allocation 크기다.

```text
STRING page 필요 용량
  = A(C) + A((N + 1) * W) + A(M(N))

fixed-width page 필요 용량
  = A(N * element_width) + A(M(N))

page end = 이 합계가 16 MiB 이하인 가장 큰 b
```

예를 들어 INT32 offsets를 쓰는 1,000,000행에 chars가 12 MiB라면,
256-byte alignment를 포함한 chars + offsets + mask 영역은 약 15.934 MiB로 16 MiB page에 들어간다.
chars만 16 MiB이면 offsets와 mask가 추가되므로 한 page에 들어가지 않는다.
`16 MiB / 평균 chars 길이`로 행 수를 결정하면 이 overhead와 구간별 길이 편차를 놓친다.

실제 저장은 한 `rmm::device_buffer` 안의 **chars → null mask → rebased offsets**다.
offsets에서 첫 chars 위치를 빼서 page 내부 첫 문자열이 0에서 시작하도록 바꾼다.
chars는 원본 column에서 D2D copy하고 mask는 첫 행의 bit가 0에 오도록 slice/copy한다.
page view와 offsets child view가 이 slab을 가리킨다.

경계 계산은 항상 mask가 있을 것으로 가정한다.
같은 컬럼이 어떤 scan에서는 mask를 갖고 다른 scan에서는 갖지 않아도 page 경계가 바뀌지 않게 하기 위해서다.
실제 RG에 null이 없으면 mask를 저장하지 않아 그 공간도 padding이 된다.

코드 근거:

- 256-byte alignment와 footprint 함수: `src/scan_manager/sirius_scan_manager.cpp:4742–4770`.
- 항상 mask-inclusive 경계: 같은 파일 `:5073–5087`.
- RG null 여부와 page-boundary offsets 묶음 조회: 같은 파일 `:5250–5277`.
- offsets rebasing과 실제 slab layout: 같은 파일 `:4808–4889`.
- 정확한 allocation과 useful bytes는 별개: `buffer = rmm::device_buffer(page_bytes, ...)`, `used_bytes = data_bytes + mask + offsets_bytes`, 같은 파일 `:4867–4869`.

기본값에서는 마지막 page에도 16 MiB를 할당하므로 남은 tail은 내부 padding이다.
`SIRIUS_PAGE_TAIL_SHRINK=1`을 별도로 켜면 RG의 마지막 page만 필요한 크기를 1 MiB 단위로 올림하여 줄일 수 있다.
이 옵션은 headline 실험의 동작으로 가정하면 안 된다.

**한 STRING 값의 chars + offsets + mask가 16 MiB를 넘으면** 현재 flat store는 해당 컬럼의 RG를 통째로 cache하지 않는다.
문자열 하나를 여러 page에 걸쳐 자르는 구현도, oversized page allocation도 이 경로에는 없다.
`row_exceeds_page` trace가 남는 위치는 같은 파일 `:5157–5167`, `:5195–5201`이다.

## 16 MiB라는 실제 allocation 증거

기존 headline 조건만 읽었다:

- ClickBench `FA_cb_{lru,ro_lru}_r{1,2}`: nonempty residency snapshot 1,920개.
- TPC-H `FA_th_{lru,ro_lru}_r{1,2,3}`: nonempty residency snapshot 1,572개.

두 workload의 모든 해당 snapshot에서 **`pages_bytes == pages * 16 * 1024 * 1024`**였다.
`chunk_bytes`, `varpage_bytes`, `pinned_fixedpage_bytes`도 모두 0이었다.
Snapshot은 warmup과 SET statement의 반복 출력도 포함하므로 실험 sample 수나 hit denominator로 쓰지 않았다.
이는 headline에서 secondary pinned store가 아닌 flat store의 16 MiB allocation을 사용했다는 근거다.

예: `experiment/FA_cb_lru_r1/celebi_fixed_variable/log_dir/sirius_2026-10-05.log:85`는
`pages_bytes=335544320`, `pages=20`, secondary bytes 모두 0을 기록한다.
335,544,320 / 20 = 16,777,216 bytes = 16 MiB다.
Population은 실제 `buffer.size()`를 accounting하고 census도 그 allocated bytes를 합산한다
(`src/scan_manager/sirius_scan_manager.cpp:5362`, `:5394–5397`).
따라서 기존 census served bytes는 chars/value payload가 아니라 **padding을 포함한 allocation traffic**이다.

## Partial page가 남아 있을 때 읽는 방식

page 경계와 row count를 저장하므로 모든 page가 resident일 필요는 없다.
부분 공급 경로는 컬럼별로 resident page의 행 구간을 모으고, 부족한 구간을 Parquet reader의
`skip_rows` / `num_rows`로 읽은 뒤 row 순서대로 column copy/concat하여 table을 만든다.
STRING과 fixed-width의 page 행 수가 달라도 이 방식으로 조립할 수 있다.
완전한 column 공급 경로는 요청한 모든 RG/page가 있어야 그 컬럼을 cache에서 가져온다.

현재 부분 공급 경로는 single-file, reader row-filter 없음, hive partition 없음일 때 적용된다.
row filtering을 post-decode로 옮기는 조건이면 unfiltered cache를 조립한 뒤 WHERE를 적용한다.
Parquet reader가 requested row range보다 큰 encoded unit을 I/O/decode할 수 있으므로,
행 구간의 decoded payload 절약이 곧 물리 NVMe read bytes 절약을 뜻하지는 않는다.

코드 근거:

- page discovery와 row provenance: `src/scan_manager/sirius_scan_manager.cpp:6192–6245`.
- partial 적용 조건: `src/op/scan/parquet_gpu_ingestible.cpp:1945–1955`.
- missing ranges 생성·read·copy/concat: 같은 파일 `:2005–2096`.
- 최종 post-filter: 같은 파일 `:2476–2487`.

## 별도 pinned-entry STRING splitter에 남아 있는 경로

`src/scan_manager/variable_width_page_index.cpp`의 secondary 구현은 현재 flat store와 구분해야 한다.
`src/legacy/`의 죽은 엔진이 아니라 **설정이나 fallback으로 사용 가능한 별도 cache 경로**다.
여기에는 평균 기반 fallback이 실제로 남아 있다.

| Secondary 경로 | 경계 계산 | allocation과 제한 |
|---|---|---|
| Slabs 기본 ON | GPU offsets에 batched `lower_bound`, row cap | chars slab을 1/2/4/8/16 MiB class로 선택 + 별도 offsets slab. 평균은 class 선택에만 사용 |
| Full offsets 옵션 | host exact offsets의 skiplist | sliced column의 exact-fit allocation. giant row는 단독 oversized page 가능 |
| Slabs OFF + alignment | fixed-width read-range의 row grid | STRING chars의 16 MiB 상한을 보장하지 않는 exact-fit column |
| Sampled byte map | 약 2,048개 offsets sample 사이를 interpolation하고 binary search | 실제 per-row footprint의 근사이며 exact-fit column |
| Sample/map 불가 fallback | `page_size / (chars_total / rows + 4)` | 평균 길이 row-count cut이 남아 있음 |

이 secondary slab 경로는 전체 page가 16 MiB인 flat store와 다르다.
기본 row cap 524,288이면 offsets allocation만 `(524288 + 1) * 4` bytes이고,
여기에 선택된 chars slab이 추가된다. Nullable/non-INT32/oversized 등으로 slab 생성에 실패하면
아래 exact/sample/grid/fallback 경로를 사용한다.
GPU `lower_bound`의 경계가 문자열 끝까지 올라가 budget을 넘길 수도 있어,
materializer의 실제 chars 크기 검증을 통과해야 slab 경로로 보관된다.

코드 근거:

- Secondary 활성화와 slabs/row cap: `src/scan_manager/sirius_scan_manager.cpp:323–397`.
- caller의 branch 및 alignment 선택: 같은 파일 `:748–828`, `:877–885`.
- slabs → exact → grid → sampled → average fallback: `src/scan_manager/variable_width_page_index.cpp:453–579`.
- device boundary search: 같은 파일 `:625–709`.
- size class 선택: 같은 파일 `:736–747`.
- null/offset/oversized 검증 및 두 allocations: 같은 파일 `:750–815`.
- plain row-byte-length helper도 oversized 단일 행 page를 허용: 같은 파일 `:47–83`.

Flat store가 이미 처리한 batch를 pinned-entry store에도 중복 populate하는 기능은 기본 OFF
(`SIRIUS_PINNED_ENTRY_CACHE`, `src/op/scan/parquet_gpu_ingestible.cpp:149–157`).
Flat store가 처리하지 못한 multi-slice fallback 등은 pinned store에 갈 수 있다.
따라서 **“headline flat page는 실제 offsets로 정확히 자른다”는 맞지만,
“현재 소스 전체에서 평균 길이 splitter가 제거됐다”는 틀리다.**

## Cache hit 그래프의 bytes 정의

새 공통 계측에서 합의한 v2 기준은 fixed-width values + STRING chars의 **decoded value bytes**다.
offsets, null masks, slab padding, 생성된 virtual columns는 제외한다.
이 정의는 partial read 여러 조각의 terminal offsets가 중복되는 문제를 피하고,
캐시가 실제 column 값을 얼마나 공급했는지 비교하게 해 준다.
Allocated bytes, H2D bytes, compressed storage I/O bytes는 각각 다른 지표다.

기존 `[cache-hit]`는 normal Parquet materialization 안에만 있어 cached-provider와 LM refetch를 빠뜨렸다.
새 계측은 `path=parquet|late_refetch|cached_provider`를 구분하고 각 completed materialize attempt를 기록해야 한다.
LM refetch는 `read_row_numbers()` early return이며, nearby row merge가 읽는 행까지 value bytes에 포함해야 한다.
HOST pin의 cached-provider `read_bytes=0`도 Parquet reader를 사용하지 않았다는 의미이고 H2D 비용은 남는다.
OOM retry가 있으면 같은 SQL의 재시도에 대한 bytes가 포함될 수 있으므로 retry와 성공 여부를 별도로 확인한다.

기존 누락 위치: `src/op/scan/gpu_ingestible.cpp:29–36`,
`src/op/scan/parquet_gpu_ingestible.cpp:645–731`, `:1760–1761`, `:2310–2314`.
