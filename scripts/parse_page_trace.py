#!/usr/bin/env python3
"""Summarise a SIRIUS_PAGE_TRACE=1 log into per-column and per-query page traffic.

The trace prints one line per page the store caches, serves, evicts, refuses or
already holds. This folds them into the numbers the two benchmarks can be compared
on: bytes served (what the cache paid off), bytes cached (what it had to fetch),
and bytes evicted (what it threw away), with reuse = served / cached.

  pixi run -e duckdb-python python scripts/parse_page_trace.py experiment/<run>
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

CACHE = re.compile(r"\[page-trace\] cache col='([^']+)' rg=(\d+) page=(\S+) rows=(\d+) bytes=(\d+)")
SERVE = re.compile(r"\[page-trace\] serve col='([^']+)' rg=(\d+) page=(\S+) bytes=(\d+)")
EVICT = re.compile(r"\[page-trace\] evict col='([^']+)' rg=(\d+) page=(\S+) bytes=(\d+) pass=(\S+)")
ALREADY = re.compile(r"\[page-trace\] already col='([^']+)' rg=(\d+) page=(\S+)")
REJECT = re.compile(r"\[page-trace\] reject col='([^']+)' rg=(\d+) reason=(\S+)")
SKIP = re.compile(r"\[page-trace\] skip col='([^']+)' reason=(\S+)")
LABEL = re.compile(r"\[page-trace\] ==== query (\S+) ====")

GB = 1024 ** 3


def parse(log: Path):
    per_col = collections.defaultdict(lambda: collections.Counter())
    per_query = collections.defaultdict(lambda: collections.Counter())
    reasons = collections.Counter()
    label = "?"
    with log.open(errors="replace") as handle:
        for line in handle:
            if "[page-trace]" not in line:
                continue
            m = LABEL.search(line)
            if m:
                label = m.group(1)
                continue
            for rx, key, size_group in ((CACHE, "cached", 5), (SERVE, "served", 4),
                                        (EVICT, "evicted", 4)):
                m = rx.search(line)
                if m:
                    col, size = m.group(1), int(m.group(size_group))
                    per_col[col][key] += size
                    per_col[col][key + "_n"] += 1
                    per_query[label][key] += size
                    per_query[label][key + "_n"] += 1
                    break
            else:
                m = ALREADY.search(line)
                if m:
                    per_col[m.group(1)]["already_n"] += 1
                    per_query[label]["already_n"] += 1
                    continue
                m = REJECT.search(line)
                if m:
                    per_col[m.group(1)]["reject_n"] += 1
                    per_query[label]["reject_n"] += 1
                    reasons[m.group(3)] += 1
                    continue
                m = SKIP.search(line)
                if m:
                    per_col[m.group(1)]["skip_n"] += 1
                    reasons[f"skip:{m.group(2)}"] += 1
    return per_col, per_query, reasons


def main() -> None:
    root = Path(sys.argv[1])
    logs = sorted(root.glob("*/log_dir/*.log"))
    if not logs:
        raise SystemExit(f"{root}: no log under */log_dir/")
    per_col, per_query, reasons = parse(logs[0])

    tot = collections.Counter()
    for counter in per_col.values():
        tot.update(counter)
    served, cached, evicted = tot["served"], tot["cached"], tot["evicted"]
    print(f"== {root.name} ==")
    print(f"  cached  {cached/GB:8.2f} GB  ({tot['cached_n']:6} pages)")
    print(f"  served  {served/GB:8.2f} GB  ({tot['served_n']:6} pages)   재사용 "
          f"{served/cached*100 if cached else 0:5.1f}%")
    print(f"  evicted {evicted/GB:8.2f} GB  ({tot['evicted_n']:6} pages)   축출 "
          f"{evicted/cached*100 if cached else 0:5.1f}%")
    print(f"  already {tot['already_n']:6}   reject {tot['reject_n']:6}   skip {tot['skip_n']:6}")
    if reasons:
        print("  거부 사유:", ", ".join(f"{k}={v}" for k, v in reasons.most_common(6)))

    print(f"\n  {'컬럼':26}{'cached':>10}{'served':>10}{'evicted':>10}{'재사용':>8}")
    rows = sorted(per_col.items(), key=lambda kv: -kv[1]["cached"])
    for col, c in rows[:14]:
        reuse = c["served"] / c["cached"] * 100 if c["cached"] else 0.0
        print(f"  {col[-25:]:26}{c['cached']/GB:9.2f}G{c['served']/GB:9.2f}G"
              f"{c['evicted']/GB:9.2f}G{reuse:7.1f}%")


if __name__ == "__main__":
    main()
