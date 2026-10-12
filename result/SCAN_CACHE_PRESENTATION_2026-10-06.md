**검증 범위.** 2026-10-06 작업 트리의 Super Sirius Parquet scan 경로와 첨부 그림을 대조했다. 엔진과 실험 설정은 수정하지 않았고 새 벤치마크는 실행하지 않았다. 구현 전체를 보여주는 그림과 headline 실험에서 활성화된 경로를 구분한다.

**판정.** GPU에 디코딩된 컬럼 페이지를 저장하고, hit에서는 재사용하고, miss에서는 압축 Parquet를 읽어 GPU에서 디코딩한다는 큰 방향은 맞다. 다만 다음 항목은 수정해야 한다.

| 첨부 그림 | 권장 수정 | 코드 근거 |
|---|---|---|
| Reordered Query | Query batch → Query sequence reorder (runner) → Execute SQL. 변경 대상은 SQL들 사이의 실행 순서다. SQL이나 연산자 순서를 이 정책이 수정하는 것은 아니다. | [러너](/home/dy1013/sirius_multi-gpu/scripts/run_random22_breakdown_var.py:238), [SQL 실행](/home/dy1013/sirius_multi-gpu/scripts/run_random22_breakdown_var.py:280) |
| Join / Aggregation / Sort | Query-plan operators (examples). 모든 쿼리가 이 세 단계를 동일 순서로 거치지 않는다. Assemble에서 나온 scan batch가 필터/프로젝션을 거쳐 이 연산자들의 입력으로 연결돼야 한다. | [scan materialize / post-filter / batch 출력](/home/dy1013/sirius_multi-gpu/src/op/scan/sirius_gpu_scan_operator.cpp:116) |
| Scan request (table, column, page) | Scan request: file / columns / row groups. Cache 내부 주소는 file / column / row group / page index. | [split](/home/dy1013/sirius_multi-gpu/src/include/op/scan/parquet_gpu_ingestible.hpp:104), [cache key](/home/dy1013/sirius_multi-gpu/src/include/scan_manager/sirius_scan_manager.hpp:430) |
| Cache Hit / Miss | Full hit / Partial hit / Miss. 일부 컬럼이나 일부 페이지가 남아 있으면 캐시 조각과 새로 읽은 조각을 합친다. | [부분 행 범위](/home/dy1013/sirius_multi-gpu/src/op/scan/parquet_gpu_ingestible.cpp:1924), [컬럼 splice](/home/dy1013/sirius_multi-gpu/src/op/scan/parquet_gpu_ingestible.cpp:2140) |
| Read from VRAM | Reuse decoded GPU pages (copy / concatenate). 이 page-store hit에는 owning-column 복사 또는 concatenate가 있으며 조립·후처리가 남는다. | [hit 서빙](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:6376) |
| Read from Host → GPU Decode | 이번 실험: NVMe/O_DIRECT → pinned HOST staging → H2D → GPU decompress/decode. HOST는 임시 경유 버퍼일 수 있으므로 읽기가 HOST 캐시 hit라는 의미는 아니다. | [device read](/home/dy1013/sirius_multi-gpu/src/io/uring/uring_reactor.cpp:577), [H2D](/home/dy1013/sirius_multi-gpu/src/io/uring/uring_reactor.cpp:1006) |
| 빠진 cache write 경로 | Assemble 후 admission / page populate, 이후 post-filter. Full hit에서는 populate를 생략한다. 캐싱이 허용되면 디코딩된 데이터를 D2D 복사로 저장한다. Budget / pressure eviction도 표시한다. | [populate 호출](/home/dy1013/sirius_multi-gpu/src/op/scan/parquet_gpu_ingestible.cpp:2321), [D2D 저장](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:4867) |
| I/O buffer: used once and discarded | Contents temporary; buffer slots reused. 데이터는 재사용용 캐시로 남기지 않지만 버퍼 allocation은 매번 버리지 않는다. H2D 완료 후 슬롯을 재사용한다. | [allocation](/home/dy1013/sirius_multi-gpu/src/io/uring/uring_reactor.cpp:501), [slot 반납](/home/dy1013/sirius_multi-gpu/src/io/uring/uring_reactor.cpp:901) |
| Prefetch Cache: compressed pages | Optional HOST compressed-file-byte chunk cache. GPU decoded page나 Parquet encoded page와 같은 단위가 아니다. 이번 headline에서는 OFF다. 켜져도 H2D와 디코딩은 남는다. | [chunk 크기](/home/dy1013/sirius_multi-gpu/src/io/cache/types.cpp:53), [HOST cache hit](/home/dy1013/sirius_multi-gpu/src/io/cache/prefetching_cache.cpp:611), [실험 YAML](/home/dy1013/sirius_multi-gpu/experiment/FA_th_ro_lru_r1/celebi_fixed_variable/sirius.yaml:1) |
| HOST secondary tier: not implemented | Planned HOST backing for decoded GPU cache pages라고 범위를 좁힌다. Flat page cache의 eviction은 discard다. 일반 실행 batch의 GPU→HOST/DISK spilling은 이미 구현돼 있다. | [page eviction](/home/dy1013/sirius_multi-gpu/src/scan_manager/sirius_scan_manager.cpp:5606), [일반 downgrade](/home/dy1013/sirius_multi-gpu/src/downgrade/downgrade_executor.cpp:170) |

**발표 그림.** [SVG](/home/dy1013/sirius_multi-gpu/result/scan_cache_architecture_verified.svg), [PDF](/home/dy1013/sirius_multi-gpu/result/scan_cache_architecture_verified.pdf), [PNG](/home/dy1013/sirius_multi-gpu/result/scan_cache_architecture_verified.png). 단순화한 Parquet 경로이며 metadata/footer 준비와 memory reservation은 생략했다. 실선은 주 실행/데이터 경로, 점선은 요청/선택 경로/캐시 갱신이다. [그림 생성 스크립트](/home/dy1013/sirius_multi-gpu/result/plot_scan_cache_architecture.py)는 분석 데이터나 GPU를 사용하지 않는다.

**약 1분 설명.**

> 먼저 batch 안의 쿼리 실행 순서를 바꿔, 같은 데이터를 사용하는 쿼리가 캐시를 재사용할 기회를 늘립니다. 각 SQL의 의미는 그대로 유지합니다. 실행할 쿼리의 scan은 파일에서 필요한 컬럼과 row group을 확인하고 GPU 페이지 캐시를 조회합니다. 이 캐시는 압축 Parquet가 아니라 이미 디코딩된 컬럼 데이터를 기본 16MiB 단위로 보관합니다. 필요한 데이터가 있으면 GPU에서 복사·조립해 사용하고, 부족한 부분은 NVMe에서 HOST 버퍼를 거쳐 GPU로 가져와 디코딩합니다. 부분 hit에서는 두 경로를 합칩니다. 새로 얻은 데이터는 admission과 메모리 예산이 허용하는 범위에서 캐시에 저장하고, 현재 쿼리의 필터와 프로젝션을 적용해 다음 연산자에 전달합니다. 따라서 목표는 다음 쿼리의 읽기·전송·디코딩을 줄이는 것입니다. 캐시 적재와 조립 비용, 메모리 압력에 따른 축출이 있기 때문에 전체 쿼리 시간이 같은 비율로 줄지는 않습니다.

**성능 그림으로 연결하는 문장.**

> 이 구조에서 paging은 디코딩된 데이터를 재사용하고, reorder는 그 데이터가 축출되기 전에 다시 사용할 기회를 바꿉니다. 반복 실행의 전체 쿼리 시간은 ClickBench에서 30.8→27.8→26.8초, TPC-H SF50에서 32.3→30.9→29.7초였습니다. ClickBench q25는 자연 순서에서 q24 뒤에 실행돼 SearchPhrase를 다시 읽지만, 재정렬하면 q24 앞에서 캐시를 재사용해 0.486→0.081초로 줄어듭니다. 다른 쿼리의 회귀와 저장 비용도 있어서 재정렬의 순이득은 전체 합으로 보고합니다.

그림 주석에는 `execution 2–3 mean; HOST compressed-byte cache OFF; TPC-H GPU cache 4GiB / ClickBench 6GiB; single GPU`를 넣는다. 이는 캐시가 유지된 반복 batch 결과다. 첫 실행을 포함한 효과와 재정렬 계산 시간이 포함된 end-to-end 지연은 별도 지표다. hit에서는 입력 data I/O·H2D·decode를 생략할 수 있지만 footer/metadata 작업 및 output 연산까지 모두 사라지는 것은 아니다.

**백그라운드 추가 조사: q24의 축출을 직접 확인.**

`FA_cb_lru_r1`, execution 2에서 q23→q24 사이 SearchPhrase live가 240→0MiB, 누적 eviction이 9072→9312MiB로 바뀌며 신규 적재는 없다. 다음 q25가 SearchPhrase 1168MiB와 EventTime 800MiB를 적재하고 SearchPhrase served는 증가하지 않는다. [q23](/home/dy1013/sirius_multi-gpu/experiment/FA_cb_lru_r1/celebi_fixed_variable/log_dir/sirius_2026-10-05.log:11925), [q24](/home/dy1013/sirius_multi-gpu/experiment/FA_cb_lru_r1/celebi_fixed_variable/log_dir/sirius_2026-10-05.log:12188), [q25](/home/dy1013/sirius_multi-gpu/experiment/FA_cb_lru_r1/celebi_fixed_variable/log_dir/sirius_2026-10-05.log:12385).

재정렬한 같은 run/execution의 q19→q25에서는 SearchPhrase 1168MiB와 EventTime 800MiB가 유지되고, 신규 적재 없이 served가 각각 같은 양만큼 증가한다. [q19](/home/dy1013/sirius_multi-gpu/experiment/FA_cb_ro_lru_r1/celebi_fixed_variable/log_dir/sirius_2026-10-05.log:11632), [q25](/home/dy1013/sirius_multi-gpu/experiment/FA_cb_ro_lru_r1/celebi_fixed_variable/log_dir/sirius_2026-10-05.log:11832). 네 표본 모두 자연 순서의 q24 종료 후 SearchPhrase live는 0이다. 특정 page ID와 allocator 원인은 추가 trace 대상이다.

**백그라운드 추가 조사: 반복 경계는 별도 실험에도 관측됨.**

아래는 첨부 headline의 세 조건과 다른, 기존 미래 인지 축출 실험이다. run horizon 조건은 각각 한 run이므로 정책의 확정 성능 비교로 쓰지 않는다. execution 1 끝→2 처음에서 다음 q8에 필요한 컬럼의 잔존 페이지와 served 카운터 차분을 읽었다. 카운터는 반복 서빙을 포함하므로 고유 페이지 수와 같지 않다.

| 기존 조건 | 경계 | q8용 잔존 페이지 | q8 served 증가 | q8 wall |
|---|---|---:|---:|---:|
| FA_th_evict_r1, execution horizon | q11→q8 | 16 | 17 | 2.318s |
| FA_th_evict_run, run horizon | q11→q8 | 104 | 105 | 1.892s |
| FA_th_ro_evict_r1, execution horizon | q19→q8 | 191 | 192 | 1.880s |
| FA_th_ro_evict_run, run horizon | q19→q8 | 246 | 247 | 1.649s |

execution horizon에서는 마지막 쿼리의 `next=`가 비지만 run horizon에서는 다음 반복 q8의 컬럼에 거리 1을 부여한다. [execution 경계 힌트](/home/dy1013/sirius_multi-gpu/experiment/FA_th_evict_r1/celebi_fixed_variable/log_dir/sirius_2026-10-05.log:14587), [run 경계 힌트](/home/dy1013/sirius_multi-gpu/experiment/FA_th_evict_run/celebi_fixed_variable/log_dir/sirius_2026-10-05.log:14588). 따라서 앞서 제안한 execution 경계 분석은 후속 조사 우선순위로 유지한다. 이 별도 조건의 숫자를 headline 막대의 효과에 섞지 않는다.
