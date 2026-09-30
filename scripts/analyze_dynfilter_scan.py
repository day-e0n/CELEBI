#!/usr/bin/env python3
"""Hot scan-latency comparison for the dynamic-filter-scan caching experiment.

Execution 1 of every run is the cold pass and is dropped; the remaining executions are
averaged per run, so what is compared is steady-state scan time per execution. Only the
`scan` bucket is reported -- the page cache can only move that one, and mixing in join or
aggregate time hides a small scan change behind unrelated noise.
"""
from __future__ import annotations
import argparse, csv, statistics
from collections import defaultdict
from pathlib import Path


def per_run_scan(bucket_csv: Path) -> tuple[float, dict[str, float]]:
    rows = [r for r in csv.DictReader(bucket_csv.open()) if int(r["execution"]) > 1]
    if not rows:
        raise SystemExit(f"{bucket_csv}: no warm executions")
    n_exec = len({int(r["execution"]) for r in rows})
    by_q = defaultdict(float)
    for r in rows:
        by_q[r["query"]] += float(r["scan"])
    return sum(by_q.values()) / n_exec, {q: v / n_exec for q, v in by_q.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", required=True,
                    help="NAME=path/to/bucket.csv[,more.csv] -- repeats of one condition")
    args = ap.parse_args()

    conds = {}
    for spec in args.label:
        name, paths = spec.split("=", 1)
        totals, per_q = [], defaultdict(list)
        for p in paths.split(","):
            t, q = per_run_scan(Path(p))
            totals.append(t)
            for k, v in q.items():
                per_q[k].append(v)
        conds[name] = (totals, per_q)

    base = next(iter(conds))
    b_mean = statistics.mean(conds[base][0])
    print(f"{'condition':<28} {'scan ms/exec':>13} {'sd':>8} {'runs':>5} {'vs base':>9}")
    for name, (tot, _) in conds.items():
        m = statistics.mean(tot)
        sd = statistics.stdev(tot) if len(tot) > 1 else 0.0
        print(f"{name:<28} {m:13.1f} {sd:8.1f} {len(tot):5} {100*(m-b_mean)/b_mean:+8.2f}%")

    names = list(conds)
    qs = sorted(set.intersection(*(set(conds[n][1]) for n in names)), key=lambda x: int(x[1:]))
    print(f"\n{'query':>6}" + "".join(f"{n[:12]:>14}" for n in names) + f"{'변화':>9}")
    for q in qs:
        vals = [statistics.mean(conds[n][1][q]) for n in names]
        chg = 100 * (vals[-1] - vals[0]) / vals[0] if vals[0] else 0
        print(f"  {q:<4}" + "".join(f"{v:14.1f}" for v in vals) + f"{chg:+8.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
