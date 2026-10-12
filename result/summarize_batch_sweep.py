#!/usr/bin/env python3
"""Summarise experiment/BATCH_SWEEP10_20261007 (scripts/run_batch_sweep.py).

Latency of a case = sum of per-query wall times, averaged over executions 2 and 3
(steady_state_workload_wall_s). Per (workload, B, condition): mean and sample std over
the complete runs. Reduction = 100 * (1 - mean CELEBI / mean Base); the paired column is
the mean and std of the per-run reduction (same run = same batch split).

  python3 result/summarize_batch_sweep.py [ROOT]
writes ROOT/batch_sweep_summary.csv and ROOT/batch_sweep_table.tex
"""
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("baseline", "paging_lru", "reorder_future")
TITLES = {"cb": "ClickBench", "th": "TPC-H SF50"}


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiment/BATCH_SWEEP10_20261007"
    rows, table = [], []
    for workload in ("cb", "th"):
        for manifest_path in sorted((root / workload).glob("b*/manifest.json"),
                                    key=lambda p: int(p.parent.name[1:])):
            manifest = json.loads(manifest_path.read_text())
            count = manifest["batch_count"]
            walls = {c: {} for c in CONDITIONS}
            pending = 0
            for case in manifest["cases"]:
                meta = json.loads(Path(case).read_text())
                if meta["status"] == "complete" and meta["comparison_valid"]:
                    walls[meta["condition"]][meta["run"]] = meta["steady_state_workload_wall_s"]
                elif meta["status"] == "skipped":
                    continue
                elif meta["status"] in ("prepared", "running"):
                    pending += 1
                else:
                    print(f"INCOMPLETE {workload} b{count} r{meta['run']} {meta['condition']}")
            if not all(walls[c] for c in CONDITIONS):
                continue
            mean = {c: statistics.mean(walls[c].values()) for c in CONDITIONS}
            paired = {c: [100 * (1 - walls[c][r] / walls["baseline"][r])
                          for r in walls["baseline"] if r in walls[c]]
                      for c in ("paging_lru", "reorder_future")}
            row = {"workload": TITLES[workload], "batches": count,
                   "runs": min(len(walls[c]) for c in CONDITIONS), "pending_cases": pending}
            for c in CONDITIONS:
                row[f"{c}_mean_s"] = round(mean[c], 3)
                row[f"{c}_std_s"] = round(sd(list(walls[c].values())), 3)
            for c in ("paging_lru", "reorder_future"):
                row[f"{c}_reduction_pct"] = round(100 * (1 - mean[c] / mean["baseline"]), 2)
                row[f"{c}_paired_reduction_mean_pct"] = round(statistics.mean(paired[c]), 2)
                row[f"{c}_paired_reduction_std_pct"] = round(sd(paired[c]), 2)
            rows.append(row)
            table.append(f"{TITLES[workload]} ($B={count}$)\n& {mean['baseline']:.2f} & "
                         f"{mean['paging_lru']:.2f} & \\textbf{{{mean['reorder_future']:.2f}}} & "
                         f"\\textbf{{{row['reorder_future_reduction_pct']:.2f}}} \\\\")
    if not rows:
        print("no complete batch count yet")
        return 0
    with (root / "batch_sweep_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "batch_sweep_table.tex").write_text("\n".join(table) + "\n")
    for r in rows:
        print(f"{r['workload']:11s} B={r['batches']} n={r['runs']} pending={r['pending_cases']}  "
              f"Base {r['baseline_mean_s']:.2f}±{r['baseline_std_s']:.2f}  "
              f"Page {r['paging_lru_mean_s']:.2f}±{r['paging_lru_std_s']:.2f} "
              f"({r['paging_lru_reduction_pct']:+.2f}%)  "
              f"CELEBI {r['reorder_future_mean_s']:.2f}±{r['reorder_future_std_s']:.2f} "
              f"({r['reorder_future_reduction_pct']:.2f}%, paired "
              f"{r['reorder_future_paired_reduction_mean_pct']:.2f}±"
              f"{r['reorder_future_paired_reduction_std_pct']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
