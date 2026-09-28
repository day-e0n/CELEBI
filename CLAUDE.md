# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Sirius is a GPU-native SQL engine that runs as a DuckDB extension, routing supported SQL
operations to the GPU (via cuDF/RMM/cuCascade) and falling back to DuckDB's CPU execution
otherwise. Once the extension is loaded it **transparently intercepts** normal SQL and runs it
on the GPU — no special syntax needed.

**The default/main branch is `dev`** (not `main`/`master`) — branch and open PRs against it.

## Build & test

Run commands through `pixi run <cmd>` (don't drop into the interactive `pixi shell`) so each
command runs in the activated environment:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=6 pixi run make # full build -- CAP THE JOBS. This is a shared box
                                           # and an unbounded `make -j` has taken it down.
pixi run make clean                        # wipe the build dir (after a failed build, before rebuilding)

pixi run make test                         # build + run the C++ unit tests (Catch2, what CI runs); make test_debug for debug

pixi run pre-commit run -a                 # all formatting/lint hooks
```

Running tests directly (non-obvious invocations):
```bash
pixi run build/release/test/unittest --test-dir . test/sql/tpch-sirius.test    # one SQLLogic file
pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[cpu_cache]"  # by Catch2 tag/test name
```
Catch2 test logs land in `build/release/extension/sirius/test/cpp/log/` — check there first when a
test fails or hangs.

**Debugging crashes/races**: `pixi run make clang-asan` / `pixi run make clang-tsan` build
sanitizer variants under `build/clang-asan/` / `build/clang-tsan/` (ASan and TSan can't be combined
in one binary). GPU-side memory errors need `compute-sanitizer` instead, not ASan/TSan. See
[docs/super-sirius/debugging.md](docs/super-sirius/debugging.md) for required `ASAN_OPTIONS` /
`TSAN_OPTIONS` and core-dump/gdb workflows.

**Python API** (links against the repo's `duckdb/` submodule via `DUCKDB_SOURCE_PATH`):
```bash
pixi run -e duckdb-python build-duckdb-python
```

**TPC-H benchmarking**: `test/tpch_performance/performance_test.py` is the canonical runner
(DuckDB CPU vs Sirius GPU, parquet or native `.duckdb` source, pinning, nsys profiling) — see
`test/tpch_performance/CLAUDE.md` for the full flag reference before writing a new benchmark script.

**Page-cache benchmarking** uses two runners with their own conditions and defaults:
`scripts/run_random22_breakdown_var.py` (TPC-H SF50) and `scripts/run_clickbench_breakdown.py`
(ClickBench). Both take `--condition baseline|celebi_fixed|celebi_variable|celebi_fixed_variable`
and `--arrival worst`; a non-baseline condition applies the reorder unless `--no-reorder`. Run one
at a time: two benchmarks sharing the NVMe inflate each other (a TPC-H baseline measured beside a
correctness check read 67.6s against 32.4s alone). Telemetry becomes numbers via
`scripts/parse_quent_operator_breakdown.py`; execution 1 is warmup, average 2 onward.

Several `SIRIUS_FIXED_PAGE_*` env vars the runners set are read by no code at all
(`..._PRUNING`, `..._VIEW_ALIGNED_SPLITS`, `..._AUTO_CACHE_ROUND_ROBIN_CHUNKS`,
`..._DEMAND_LOAD`) — grep before attributing any behaviour to one.

**Correctness before performance**: `scripts/check_clickbench_correctness.py` and
`scripts/check_tpch_correctness.py` run every query against DuckDB on the full dataset and compare
values. TPC-H matches exactly. ClickBench has nine queries whose results differ because
`ORDER BY ... LIMIT` leaves ties unordered — the aggregates match, the chosen rows do not; check
the ordering keys before calling such a difference a bug.

`SIRIUS_PAGE_TRACE=1` logs every page the store caches, serves, evicts or refuses (with the
reason), and `SET page_trace_label='...'` marks which statement the lines belong to.

**Worktrees**: submodules are not auto-initialized — after creating one, run
`git submodule update --init --recursive`.

**Not part of the main C++/CUDA build**: `rust/` (Cargo workspace: Quent telemetry
model/bridge/analyzer/server, `pixi run quent` to serve the UI) and `experimental/` (e.g. a
StarRocks integration under its own `pixi.toml`, gated by a separate CI job) are independent
workspaces — don't assume root `pixi run make` touches them.

## Working on this repo

**Do not change the settings of a running experiment without being asked.** Cache budgets,
conditions, skip lists, reorder policies, min-free floors: these are the experiment. Changing one
to make a run finish, or to get a nicer number, invalidates the comparison and wastes the GPU
hours that produced it. If a configuration will not complete, say so and report that -- it is a
result. Past incidents: a variable-width budget silently set to 1GB when the defaults are 4GB and
2GB; a ClickBench budget cut from 6GB to 2GB because 6GB was hitting OOM; the separate variable
cache switched off mid-comparison.

**Measure before claiming a cause.** Several plausible explanations in the page-cache work were
wrong and were only caught by instrumenting: "joins prevent caching", "column reuse differs",
"pruning is the confound" (the env var it named is not read by any code), "URL has no complete
row group" (it had two). State what was measured and how; if something is a guess, say it is a
guess.

**A requirement stated once stays in force.** If the ask is "cut pages on the offsets, not on an
average", that applies to every path that cuts pages, including ones written later. Re-read the
ask before reporting something as done.

**Use the project's words.** The unit is a *page*; a *row group* is a row group. Do not coin new
terms for things that already have names.

## Architecture

**Super Sirius** is the live engine: namespace `sirius`, source under `src/op/` (operators),
`src/planner/` (plan builders + `sirius_physical_plan_generator.cpp`), `src/pipeline/`,
`src/cuda/` (GPU kernels). **Read `docs/super-sirius/` before modifying Super Sirius code** —
see its [README](docs/super-sirius/README.md) for reading order.

**Everything under `src/legacy/` is the dead `gpu_processing` path — do not modify it.** All new
work targets Super Sirius. Memory spilling / CPU fallback is handled by the downgrade executor
(`src/downgrade/`, `src/creator/`); see `docs/super-sirius/memory-management.md`.

**Every byte lives in exactly one memory tier** — `GPU` → `HOST` (NUMA-local pinned memory) →
`DISK` — and the downgrade executor moves data down under memory pressure, back up on demand.
Sirius runs **multi-GPU on a single node** (one process pins every visible GPU via
`CUDA_VISIBLE_DEVICES`; there is no distributed/multi-node execution in this codebase) — see
`docs/super-sirius/multi-gpu-architecture.md` before touching cross-GPU data placement,
scheduling, or dynamic filters.

Before implementing operators / memory / expression / I/O work, run `/module-context <task>` to
load accurate cudf/rmm/duckdb/cucascade API docs.

## Usage

Load the extension and run normal SQL — Sirius intercepts it transparently and runs supported
queries on the GPU (controlled by the `gpu_execution` setting, on by default):

```sql
LOAD 'build/release/extension/sirius/sirius.duckdb_extension';
SELECT ...;                  -- transparently routed to the GPU
-- SET gpu_execution = false;  -- to disable interception
```
