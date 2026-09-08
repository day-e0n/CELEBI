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
