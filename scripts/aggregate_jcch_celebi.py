#!/usr/bin/env python3
"""Aggregate the JCC-H SF50 baseline-vs-CELEBI runs across repeats.

parse_quent_operator_breakdown.py emits one bucket CSV per run; this rolls the
repeats of each condition into a single comparison. The first execution of every
repeat is the cold pass and is dropped (the same warmup rule
build_operator_breakdown_comparison.py applies), then the remaining executions of
all repeats are pooled -- so the spread reported per condition is the spread
across repeats, not across warm executions within one repeat.
"""
from __future__ import annotations

import argparse, csv, statistics
from collections import defaultdict
from pathlib import Path

BUCKETS = ("scan", "join", "aggregate", "filter", "sort", "other")


def load(path: Path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def repeat_totals(paths: list[Path]) -> tuple[list[dict], dict]:
    """Per-repeat workload totals, plus the pooled per-query bucket means."""
    per_repeat, pooled = [], defaultdict(list)
    for p in paths:
        rows = [r for r in load(p) if int(r["execution"]) > 1]  # drop the cold pass
        if not rows:
            raise SystemExit(f"{p}: no warm executions (need execution > 1)")
        tot = defaultdict(float)
        by_qe = defaultdict(lambda: defaultdict(float))
        for r in rows:
            for b in BUCKETS:
                tot[b] += float(r[b] or 0)
                by_qe[r["query"]][b] += float(r[b] or 0)
        n_exec = len({int(r["execution"]) for r in rows})
        # Per-execution average, so repeats with different execution counts compare.
        per_repeat.append({b: tot[b] / n_exec for b in BUCKETS} | {"n_exec": n_exec})
        for q, bb in by_qe.items():
            pooled[q].append({b: bb[b] / n_exec for b in BUCKETS})
    return per_repeat, pooled


def mean_sd(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True, type=Path)
    ap.add_argument("--proposed", nargs="+", required=True, type=Path)
    ap.add_argument("--proposed-label", default="celebi_fixed_variable")
    ap.add_argument("--out-csv", type=Path)
    args = ap.parse_args()

    base_rep, base_q = repeat_totals(args.baseline)
    prop_rep, prop_q = repeat_totals(args.proposed)

    print(f"{'bucket':<10} {'baseline ms':>14} {'':>8} {args.proposed_label+' ms':>22} {'':>8} {'change':>9}")
    out = []
    for b in BUCKETS + ("TOTAL",):
        bv = [sum(r[x] for x in BUCKETS) if b == "TOTAL" else r[b] for r in base_rep]
        pv = [sum(r[x] for x in BUCKETS) if b == "TOTAL" else r[b] for r in prop_rep]
        bm, bs = mean_sd(bv)
        pm, ps = mean_sd(pv)
        chg = (pm - bm) / bm * 100 if bm else 0.0
        print(f"{b:<10} {bm:14.1f} {'±'+format(bs,'.1f'):>8} {pm:22.1f} {'±'+format(ps,'.1f'):>8} {chg:+8.1f}%")
        out.append({"bucket": b, "baseline_ms": f"{bm:.3f}", "baseline_sd": f"{bs:.3f}",
                    "proposed_ms": f"{pm:.3f}", "proposed_sd": f"{ps:.3f}", "pct_change": f"{chg:.2f}"})

    print(f"\n반복 {len(base_rep)}회 (baseline) / {len(prop_rep)}회 ({args.proposed_label}), "
          f"실행당 평균, 각 반복의 execution 1은 cold로 제외")

    print(f"\n{'query':>6} {'baseline ms':>12} {'proposed ms':>12} {'change':>9}   scan만")
    for q in sorted(set(base_q) & set(prop_q), key=lambda x: int(x[1:])):
        bt = statistics.mean(sum(r[x] for x in BUCKETS) for r in base_q[q])
        pt = statistics.mean(sum(r[x] for x in BUCKETS) for r in prop_q[q])
        bsc = statistics.mean(r["scan"] for r in base_q[q])
        psc = statistics.mean(r["scan"] for r in prop_q[q])
        sc = (psc - bsc) / bsc * 100 if bsc else 0.0
        print(f"  {q:<4} {bt:12.1f} {pt:12.1f} {(pt-bt)/bt*100 if bt else 0:+8.1f}%   {sc:+7.1f}%")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader(); w.writerows(out)
        print(f"\nwrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
