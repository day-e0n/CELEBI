"""Offline telemetry analysis of the exact three headline conditions.

Run with pixi run -e duckdb-python python result/analyze_headline_task_occupancy.py.
Computing spans include host waits and CUDA enqueue time, not just GPU kernels.
reserve_slot_share_s integrates reserving tasks against unused worker slots, divided
by four. It is a scheduling diagnostic, not measured or causal wall time lost.
"""
import collections
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'result/tpch_sf50_per_query.py').read_text().split('FIELDS = ')[0]
helper = {'__file__': str(ROOT / 'result/tpch_sf50_per_query.py')}
exec(compile(source, 'existing_task_span_helpers', 'exec'), helper)
rows = []
for bench, reps in [('th', range(1, 4)), ('cb', range(1, 3))]:
    for condition in ['base', 'lru', 'ro_lru']:
        for repetition in reps:
            run = f'FA_{bench}_{condition}_r{repetition}'
            spans = helper['task_spans'](run)
            walls = helper['walls'](run)
            expected = 22 if bench == 'th' else 41
            for execution in [2, 3]:
                assert sum(e == execution for q, e in walls) == expected, run
            for (query, execution), wall in walls.items():
                pieces = spans[(query, execution)]
                part = helper['split'](pieces)
                scan = sum(part[k] for k in ['scan_full', 'scan_resv', 'scan_idle'])
                computing_task_s = sum((b-a)/1e9 for a,b,k in pieces if k in ['scan','compute'])
                rows.append(dict(bench=bench, condition=condition, repetition=repetition,
                    execution=execution, query=f'q{query}', wall_s=wall,
                    scan_active_s=scan, scan_slots_full_s=part['scan_full'],
                    scan_with_reserving_s=part['scan_resv'],
                    scan_with_idle_slots_s=part['scan_idle'],
                    other_only_s=part['other_ops'],
                    outside_computing_s=wall-scan-part['other_ops'],
                    reserve_slot_share_s=part['resv_lost_s'],
                    computing_task_s=computing_task_s))
out = ROOT / 'result/headline_three_conditions_task_occupancy.csv'
with out.open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summary = []
for bench in ['th','cb']:
    for condition in ['base','lru','ro_lru']:
        selected=[r for r in rows if r['bench']==bench and r['condition']==condition]
        samples=collections.defaultdict(collections.Counter)
        for r in selected:
            for k,v in r.items():
                if k.endswith('_s'): samples[r['repetition'],r['execution']][k]+=v
        avg={k:statistics.mean(s[k] for s in samples.values()) for k in next(iter(samples.values()))}
        union=avg['scan_active_s']+avg['other_only_s']
        avg['mean_tasks_while_computing']=avg['computing_task_s']/union
        summary.append(dict(bench=bench, condition=condition, samples=len(samples), **avg))
        print(bench, condition, {k:round(v,4) for k,v in avg.items()})
with (ROOT / 'result/headline_three_conditions_task_summary.csv').open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
print(out)
