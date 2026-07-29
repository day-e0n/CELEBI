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
pixi run make                              # full build (uses all cores)
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

**Worktrees**: submodules are not auto-initialized — after creating one, run
`git submodule update --init --recursive`.

**Not part of the main C++/CUDA build**: `rust/` (Cargo workspace: Quent telemetry
model/bridge/analyzer/server, `pixi run quent` to serve the UI) and `experimental/` (e.g. a
StarRocks integration under its own `pixi.toml`, gated by a separate CI job) are independent
workspaces — don't assume root `pixi run make` touches them.

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
