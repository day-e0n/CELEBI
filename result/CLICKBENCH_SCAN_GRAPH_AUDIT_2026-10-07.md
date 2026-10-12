# ClickBench scan graph audit

The requested query-time-to-scan conversion had two distinct sources. The original
query-time figure used A_* runs and excluded q28/q29 (41 queries). The subsequently
displayed scan graph instead used the completed FULL_CACHE_PIN latency cases and
included all 43 queries. Those figures must not be treated as a metric-only
conversion of the same dataset.

All values below exclude execution 1 and average executions 2 and 3.
The A_* ClickBench reference additionally averages independent runs 1 and 2.
The full dataset has one independent run per condition.

| Source / scan work sum (s) | Baseline | Paging LRU | Reorder + future eviction |
|---|---:|---:|---:|
| A_* original 41-query scope | 70.455849 | 54.725354 | 50.952253 |
| FULL_CACHE_PIN all 43 | 153.099315 | 121.365978 | 125.029372 |
| FULL_CACHE_PIN q28/q29 contribution | 83.354770 | 65.618165 | 78.154865 |
| FULL_CACHE_PIN remaining 41 queries | 69.744545 | 55.747813 | 46.874507 |

The remaining-41 subset is a diagnostic of the full run: q28/q29 were actually
executed and could change cache state. It is not equivalent to rerunning a workload
that excludes them.

For q28, scan work is 31.982402 / 15.233354 / 27.121629 s. For q29 it is
51.372368 / 50.384811 / 51.033237 s. The full-run reversal between paging
and reorder therefore arises in the q28/q29 contribution, predominantly q28.
This locates the difference; it does not establish a policy or memory-pressure
cause. Reorder q28's execution 2/3 scan work values are 18.724491 / 35.518767 s.
One run and two measured executions are insufficient for a stable general conclusion.

## Independent verification

`audit_clickbench_scan_graph.py` reconstructs raw Task Computing intervals and
checks all 258 steady-state samples against the plotted CSV. Every attributed scan
kind in the ClickBench dataset is GPU_SCAN. It also parses independent scan-stage
logs, records actual OOM events, and computes the union of overlapping scan intervals.

q28 scan-stage sums (s) are 31.981865 / 15.152089 / 27.048690, close to the
telemetry sums; this supports the existence of the difference in original data.
q29 logs include an OOM at PROJECTION in one measured reorder execution. No q28
OOM-at-operator event is recorded in the measured executions. This is not enough
to assign a cause to the q28 difference.

The scan graph sums task work, including overlapping intervals and work/waits
within Computing. It is not scan wall latency. For example baseline q28's scan
work is 31.982402 s, union of scan intervals 16.640642 s, and query wall time
16.731426 s. Union measures time with at least one scan task in Computing; it
does not isolate GPU kernel activity, and does not include all scheduling gaps.

Full-workload query wall sums are 87.467314 / 82.516849 / 80.542807 s,
so the scan-work reversal does not imply a query-wall reversal.

## Artifacts

- Original figure's same-source scan conversion: `cumulative_scan_work_reference.{png,svg,pdf,csv}`.
- Full 43-query scan figure: `cumulative_scan_work_full.{png,svg,pdf,csv}`.
- Independent per-execution audit: `clickbench_scan_graph_audit.csv`.
- Raw samples, operator kinds, telemetry hashes and layout dimensions accompany both plots.

No experiment settings or engine code were changed during this audit.
