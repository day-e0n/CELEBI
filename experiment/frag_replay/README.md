# How the 6 GiB cache budget is spent: batch vs exact-size pages vs uniform 16 MiB pages

Replays the scan sequence of a real ClickBench run (41 queries x 3 executions, baseline,
`experiment/SL_base`) against a `cudaMallocAsync` pool capped at 6 GiB, under three cache
granularities, and records after every scan where the budget went.

| file | role |
|---|---|
| `sizes.py` | decoded bytes per (column, row group) from the parquet file + the scan sequence from the log -> `sizes.json` |
| `gen_trace.py` | allocation traces for `batch`, `page`, `slab` |
| `replay.cu` | replays a trace on the capped pool; LRU eviction; probes the largest placeable block after each scan -> `<policy>.csv` |
| `plot.py` | `budget_composition.png`, `budget_over_time.png`, `summary.csv` |

Run (GPU 1, idle): build `replay.cu` inside `pixi run` (`nvcc -ccbin $CXX`), then
`./replay trace_<policy>.txt 6442450944 <policy>.csv` for each policy, then `plot.py`.

**Definitions.** External fragmentation = free bytes in the capped pool minus the largest
single allocation the pool can still place (found by trying, 2 MiB resolution). Padding =
bytes allocated to a slab but not holding data. "Extra eviction from fragmentation" = bytes
evicted after the requested bytes already fit the budget but the pool could not place the
allocation.

**Policies.** `batch`: one entry per (scan batch, column), held as cudf does (data, null mask,
STRING offsets as separate allocations). `page`: the design before uniform slabs, <=16 MiB of
data per page with mask/offsets as separate allocations. `slab`: the current design, one
16 MiB allocation per page.

**Limits.** The pool is dedicated to the cache; in Sirius the cache shares the query pool.
Sizes are whole row groups (filters ignored) and a hit is key identity only, so hit counts are
an upper bound on reuse, not Sirius's. STRING page boundaries use the row group's average row
length.

Result (average after 50 warm-up scans, GiB of 6; served/copied are totals over the run, which
asks for 503.5 GiB):

| | reused data | not-yet-reused data | internal frag (padding) | external frag | free usable | extra eviction from frag | served from cache | copied into cache |
|---|---|---|---|---|---|---|---|---|
| batch | 1.29 | 4.08 | 0 | 0.34 | 0.29 | 257 | 55.6 | 448 |
| exact pages | 1.63 | 3.53 | 0 | 0.75 | 0.10 | 408 | 90.7 | 413 |
| 16 MiB slabs | 1.63 | 3.79 | 0.53 | 0 | 0.05 | 0 | 90.7 | 413 |

Hit *counts* are not comparable across designs (a batch hit is one large entry, a page hit one
16 MiB page); compare served bytes. 92-94% of evicted bytes were never served in every design.

## A -> B -> C (fresh run, 2026-10-02)

A = q16 (UserID), B = q17 (UserID, SearchPhrase), C = q19 (UserID, EventTime, SearchPhrase),
run on Sirius in that order (`experiment/ABC_base`, cache off, prefetch off) to record the real
data batches: A grouped row groups [0-5],[6-9]; B in pairs; C mostly one row group per batch.
`abc_trace.py` makes the traces (batch = one entry per whole data batch, keyed by columns and
row groups; page = uniform 16 MiB pages), `replay.cu` measures them on the 6 GiB pool,
`abc_plot.py` draws `abc_budget.png`.

| after | batch: held / duplicate / served so far | 16 MiB page: held / padding / served so far |
|---|---|---|
| A | 0.76 / 0 / 0 | 0.76 / 0.02 / 0 |
| B | 2.59 / 0.76 / 0 | 1.83 / 0.09 / 0.76 |
| C | 5.17 / 2.59 / 0 | 2.59 / 0.12 / 2.59 |

Nothing is evicted in this run, so external fragmentation is zero for both designs; the
difference is duplication and reuse.

## Eight queries, two rounds (fresh run, 2026-10-02)

Order A..H = q16, q17, q19, q9, q10, q33, q34 (URL, 8.8 GiB), q36, run twice on Sirius
(`experiment/SEQ8_base`, cache off, prefetch off); 37.1 GiB read in all. `seq_trace.py` builds
traces for batch / exact-size pages / 16 MiB pages, `replay.cu` measures them on the 6 GiB pool
(also dumping resident entries per query so duplicates can be counted), `seq_plot.py` draws
`seq_budget.png` and `seq_summary.csv`.

| | served from cache | copied into cache | extra eviction from fragmentation | max duplicate held | max external frag | max padding |
|---|---|---|---|---|---|---|
| batch | 0.0 GiB | 37.2 GiB | 8.2 GiB | 2.89 GiB | 0.35 GiB | 0 |
| exact-size pages | 9.8 GiB | 27.4 GiB | 22.4 GiB | 0 | (not plotted) | 0 |
| 16 MiB pages | 9.8 GiB | 27.4 GiB | 0 | 0 | 0 | 0.67 GiB |

Pages serve 4.7 GiB in round 1 and 5.1 GiB in round 2: reuse happens within a round (B reuses
A's UserID, C reuses A and B), not across rounds, because G's 8.8 GiB URL scan pushes everything
else out of the 6 GiB cache under LRU.
