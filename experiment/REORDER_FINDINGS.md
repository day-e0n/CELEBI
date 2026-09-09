# Page-cache query reordering on TPC-H SF100

fixed+variable page cache, 6 GB fixed / 4 GB variable, column-keyed entries,
`celebi_fixed_variable`, execution 1 discarded as warmup.

## Measured orders

| order | total (s) | n | vs overlap reorder |
|---|---|---|---|
| no cache (baseline) | 219.7 ± 1.3 | 6 | +18.1% |
| shared-per-cost | 207.2 ± 1.4 | 3 | +11.4% |
| q1 moved to position 1 | 195.5 ± 6.0 | 3 | +5.1% |
| **overlap reorder (current CELEBI)** | **186.0 ± 3.3** | 4 | — |
| random arrival | 182.9 ± **17.3** | 4 | -1.6% |
| cost-ascending, from plan estimate | 181.1 ± 4.7 | 3 | -2.6% |
| cost-descending (control) | 180.6 ± 7.7 | 3 | -2.9% |
| byte-weighted adjacent overlap | 179.2 ± 11.6 | 3 | -3.7% |
| widest-opener-first | 177.9 ± 4.3 | 3 | -4.3% |
| shared-columns-first | 177.5 ± 4.8 | 3 | -4.6% |
| **cost-ascending, from measured times** | **171.9 ± 9.4** | 6 | **-7.6%** |

## What the overlap reorder actually does

It does not improve the mean (random arrival 182.9 -> 186.0). It removes the
dependence on arrival order: all four random inputs produce the SAME output
order, so the spread collapses from ±17.3 to ±3.3. It rescues a bad arrival
(207.5 -> 181.7, +12.4%) and damages a good one (167.1 -> 188.7, -12.9%).

The defensible claim is bounded worst case, not average speedup.

An earlier comparison used a hand-built "worst" arrival order and concluded the
reorder was a loss. That baseline was not neutral -- `worst_case_sequence()`
seeds with the query holding the most columns. Random arrival orders are the
neutral baseline; that correction is what moved the reorder from "-6.1% loss"
to "no mean change, 5x less variance".

## How the cache actually works (measured)

22 solo runs, one query each, reading the engine's own entry names:

- **16 entries, keyed by (file, pushed-down filter).** 8 are unfiltered and 8
  carry a predicate (`lineitem;filter=((#1 <`, `orders;filter=((#2 >=`, ...), so
  filters DO fragment the cache. An earlier version of this file claimed 7
  entries, all unfiltered, on the grounds that `disable_filter_pushdown` is set
  whenever auto-caching is on; that is wrong twice over. The flag is
  `cache_before_filter_enabled() && fixed_page_auto_cache_enabled()`, and
  `SIRIUS_FIXED_PAGE_CACHE_BEFORE_FILTER` is not set by the runners, so pushdown
  stays ON. The "7 unfiltered" count came from reading only
  `auto_cache_populate_direct` lines, which do not cover every entry a query
  touches.
- **An entry holds a scan's projection AND filter columns.** q6 projects 2
  lineitem columns and caches 4. Reproduced on 26 of 29 (query, table) pairs;
  the 3 misses are scans carrying a dynamic filter, which are skipped entirely.
- **Entries accumulate and freeze at the per-entry ceiling** (budget/2 = 3 GB);
  `admission_stop_widening` fires 476-732 times per run. A single SF100 lineitem
  column is 4.47 GB decoded (600M rows x 8 B), so not even one column fits whole.

## What predicts total time (n=14 orders)

| signal | r |
|---|---|
| `using` (cache hits) | **-0.689** |
| `auto_cache_populate_direct` | **+0.728** |
| stop_widening | -0.262 |
| lineitem first-populate column count | -0.159 |
| max entry bytes | +0.098 |

Hits determine time. Entry width does not -- the "open the entry with the widest
query" hypothesis was tested directly (widest-first 177.9, q1-first 195.5) and
is not supported across the 14 orders.

## What does NOT predict (the open problem)

Hits determine time, but no offline model predicts hits. Four attempts, scored
against 14 measured orders:

| model | vs measured hits | vs total time |
|---|---|---|
| global LRU over (table, column), column-counted | — | +0.00 |
| byte-weighted, LRU-simulated | — | (order-level -0.41) |
| entries split by EXPLAIN filter signature | — | -0.13 |
| entries per table, all-or-nothing coverage, 3 GB freeze | **+0.245** | +0.363 |

The last is structurally closest to the engine and still explains almost
nothing. Every overlap-shaped objective tried (column count, byte-weighted
adjacency, LRU-simulated residency, shared-coverage density) lands between
177 and 207 s with no ordering that tracks the model's score.

The one order that reliably wins, cost-ascending at 171.9 s, needs measured
per-query scan times -- it is an oracle. Recomputed from plan-estimated bytes
(Spearman +0.63 against measured) it recovers only 5 s of the 14 s:
181.1 vs 186.0. Closing that gap is the open work.

## What decides the cache's benefit: not its share of the data

Three benchmarks, same 6 GB fixed / 4 GB variable budget, no reordering:

| benchmark | decoded | cache holds | fixed | fixed+variable |
|---|---|---|---|---|
| TPC-H SF50 | 52.2 GB | 19.2% | -14.6% | **-16.3%** |
| TPC-H SF100 | 104.4 GB | 9.6% | -9.6% | **-20.9%** |
| ClickBench | 61.5 GB | 16.3% | -12.1% | **-35.7%** |

SF50 was run to test the obvious explanation for ClickBench's much larger gain:
that a 10 GB cache simply covers more of a smaller dataset. It does not hold.
SF50 is smaller than ClickBench and its cache covers a LARGER share of it
(19.2% vs 16.3%), yet it gains less than half as much (-16.3% vs -35.7%).

What separates them is how much of the workload's scanning the cache is allowed
to serve. ClickBench is one denormalised table with no joins, so no scan ever
carries a dynamic filter and every scan is a cache candidate. On TPC-H, 1032 of
1165 cache skips per run are `dynamic_filter_scan` -- the cache is shut out of
89% of scans before capacity is even consulted.

A second reading of the same table: the variable-width cache adds -1.7 points on
SF50, -11.3 on SF100 and -23.6 on ClickBench. On the smallest dataset the
fixed-width cache already holds what the workload re-reads, and STRING paging
has little left to contribute.

## Dynamic-filter scans: why the read gate cannot simply be opened

Serving a dynamic-filter scan from cache is *correct* -- DYNAMIC_FILTER is a
separate operator above the scan, so cached batches are masked exactly like
decoded ones -- and it was measured at 173.7 -> 191.4 s (+10.2%), scan +14.4 s.

The cost is rows, not bytes. `disable_filter_pushdown` is gated on
`SIRIUS_FIXED_PAGE_CACHE_BEFORE_FILTER`, which the runners do not set, so
pushdown stays ON and the dynamic filter reaches the reader, where it prunes row
groups by statistics and drops rows during decode. Serving from cache bypasses
all of it: q12 goes 992 ms -> 7538 ms against a 7697 ms no-cache baseline, i.e.
straight back to a full scan. Per-query, 9 queries gain 7.7 s and 13 lose 22.2 s.

The proper fix is prune-first-then-serve, and the machinery exists
(`set_cached_row_groups`, `parquet_gpu_ingestible.cpp:696`, which subtracts the
cache-served row groups from what the reader reads). Two things block it:

1. **The subtraction happens before pruning**, so cache-served row groups never
   reach `filter_row_groups_with_stats` at all. The comment there claims the two
   steps compose; they do not.
2. **A dynamic filter has no value at cache-assignment time.**
   `scan_manager_->prepare_for_query` (which runs `try_assign_cached_entries` and
   builds the provider) is called from `sirius_context.cpp:678`, before any
   execution; the filter is published by the join's build side during the query.
   Fixing (1) alone changes nothing, because the provider -- built earlier --
   still hands over every chunk it holds.

So the fix requires moving cache assignment from query setup to scan activation,
which also reopens the race the provider's constructor comment documents
(materialising every batch under `_pinned_entries_mutex` to avoid a concurrent
insert reallocating the vectors it reads -- previously seen as intermittent
duplicate rows). Against a measured ceiling of -4.4%, that was judged not worth
it; the read gate is split out as
`SIRIUS_FIXED_PAGE_CACHE_DYNAMIC_FILTER_REUSE` and defaults to closed.

The static-filter path has no equivalent problem, and this was checked rather
than assumed: with the cache on, **22 of 22 SF100 queries are faster than
baseline and none is slower** (-41.8 s in total). Static predicates are part of
the cache key, so a filtered scan usually meets an entry holding exactly its
filtered rows and there is no pruning left to lose -- q12 is 7697 -> 992 ms, the
mirror image of what the dynamic read gate does to it.

## Files

- `scripts/probe_column_bytes.py` -- per-column DECODED bytes from parquet.
  Deliberately not `total_uncompressed_size`: that is the encoded size, and on
  SF100 the two differ by 16x for `l_discount` (11 distinct values: 0.28 GB
  encoded, 4.47 GB of int64 in VRAM). Ranking a cache by encoded size prefers
  exactly the columns cheapest to re-decode.
- `scripts/probe_scan_requests.py` -- per (query, table) column sets from EXPLAIN.
- `scripts/entry_cache_model.py` -- the entry-semantics simulator.
- `scripts/cache_aware_query_reorder.py` -- policies `fixed-overlap` (original),
  `byte-overlap`, `byte-lru`, `cost-ascending`.
- `experiment/colbytes_sf100_decoded.json`, `engine_entries_sf100.json`,
  `query_table_columns_sf100.json`.

## Dynamic-filter scans (separate axis, closed)

1032 of 1165 cache skips per run are `dynamic_filter_scan`. The gate is not a
routing problem -- DYNAMIC_FILTER is a separate operator above the scan, so
cached batches are masked exactly like decoded ones. Both directions were
measured:

- write gate open (SF50, prior work): hits 88 -> 37, scan +8.8%.
- both gates open (SF100): OOM at the same query in 3 of 3 runs -- caching a
  probe-side lineitem scan means caching every row the filter would discard.
- read gate only (added here, `SIRIUS_FIXED_PAGE_CACHE_DYNAMIC_FILTER_REUSE`):
  173.7 -> 191.4 s (+10.2%), scan +14.4 s. Serving a dynamic-filter scan from
  cache bypasses the reader, where the dynamic filter does row-group pruning
  and row filtering (`parquet_gpu_ingestible.cpp:1185`). It costs rows, not
  bytes, which is why the read direction is not free either.

Row-group-level partial residency already exists (`set_cached_row_groups`,
`parquet_gpu_ingestible.cpp:696`), so prune-then-serve-survivors is the
untried option.
