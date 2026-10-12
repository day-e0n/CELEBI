"""Why the page cache is (not) used: per-column lookup outcomes and the scan time breakdown.

Needs a run with SIRIUS_SCAN_CACHE_TRACE=1 (and ideally SIRIUS_PAGE_TRACE=1), e.g.
experiment/SC_trace. Optionally a baseline run of the same order for the time comparison.

  python3 analyze_scan_cache.py SC_trace [baseline_run]
"""
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
GIB = 2 ** 30
d = json.load(open(HERE / "sizes.json"))
rg_rows = {int(k): v for k, v in d["rg_rows"].items()}
mask = lambda r: ((r + 511) // 512) * 64


def rg_bytes(col, rg):
    """Decoded bytes of one column in one row group (data + null mask + STRING offsets)."""
    info = d["cols"].get(col)
    if info is None:
        return 0
    r = rg_rows[rg]
    if info["string"]:
        return d["chars"].get(f"{col}|{rg}", 0) + 4 * (r + 1) + mask(r)
    return r * info["width"] + mask(r)


def log_of(run):
    return glob.glob(str(HERE.parent / run / "*" / "log_dir" / "*.log"))[0]


CACHE = re.compile(r"\[scan-cache\] file='[^']*' col='([^']*)' filtered=(\d) rgs=(\d+) full=(\d+) "
                   r"partial=(\d+) predicate=(\d+) evicted=(\d+) declined=(\d+) never=(\d+) full_bytes=(\d+)")
DECIDE = re.compile(r"\[scan-cache-decision\] file='[^']*' row_groups=(\d+) columns=(\d+) filtered=(\d) "
                    r"available=(\d+) served=(\d+) full_hit=(\d)")
AUDIT = re.compile(r"\[scan-audit\] parquet_materialize .*columns=(\S*) row_groups=(\S*) .*"
                   r"duration_us=(\d+) lookup_us=(\d+) materialize_us=(\d+) read_us=(\d+) "
                   r"assemble_us=(\d+) lock_wait_us=(\d+) stage_prep_us=(\d+) full_hit=(\d)")
POPULATE = re.compile(r"\[scan-populate\] store=pages us=(\d+)")
STAGE = re.compile(r"\[scan-stage\] materialize_us=(\d+) postfilter_us=(\d+) make_batch_us=(\d+)")


def scan_times(run):
    t = defaultdict(float)
    n = 0
    for line in open(log_of(run), errors="replace"):
        m = AUDIT.search(line)
        if m:
            n += 1
            for k, v in zip(("total", "lookup", "materialize", "read", "assemble", "lock_wait",
                             "prep"), m.groups()[2:9]):
                if k == "lock_wait":  # a running total, not a per-scan figure
                    t[k] = max(t[k], int(v) / 1e6)
                else:
                    t[k] += int(v) / 1e6
            continue
        m = POPULATE.search(line)
        if m:
            t["populate"] += int(m[1]) / 1e6
            continue
        m = STAGE.search(line)
        if m:
            t["postfilter"] += int(m[2]) / 1e6
            t["make_batch"] += int(m[3]) / 1e6
    t["scans"] = n
    return t


run = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else None

# ---- 1. lookup outcomes, in bytes of decoded column data ------------------------------
# A column lookup row counts per-status row groups; the bytes are the row groups' decoded
# sizes. Ordering inside a line is unknown (which row groups were full), so per-status bytes
# use the column's average row-group size, except full_bytes which the store reports.
status = defaultdict(float)
lost_partial_cols = 0
lookups = 0
for line in open(log_of(run), errors="replace"):
    m = CACHE.search(line)
    if not m:
        continue
    col, filtered, rgs = m[1], int(m[2]), int(m[3])
    full, partial, pred, evict, decl, never = map(int, m.groups()[3:9])
    lookups += 1
    if col not in d["cols"] or rgs == 0:
        continue
    avg = sum(rg_bytes(col, g) for g in rg_rows) / len(rg_rows)
    if full == rgs:
        status["served: every row group cached"] += full * avg
    elif full > 0:
        status["lost: some row groups cached, column read whole"] += full * avg
        lost_partial_cols += 1
    status["miss: page run incomplete"] += partial * avg
    status["miss: filter differs"] += pred * avg
    status["miss: evicted earlier"] += evict * avg
    status["miss: declined at admission"] += decl * avg
    status["miss: never cached before"] += never * avg
    if 0 < full < rgs:
        pass

total = sum(status.values())
print(f"== column lookups: {lookups}, requested {total / GIB:.1f} GiB of decoded column data")
for k, v in sorted(status.items(), key=lambda kv: -kv[1]):
    print(f"  {k:52} {v / GIB:7.1f} GiB  {100 * v / total:5.1f}%")
print(f"  columns read whole although some row groups were cached: {lost_partial_cols}")

# ---- 2. scan-level decisions ------------------------------------------------------------
dec = defaultdict(int)
for line in open(log_of(run), errors="replace"):
    m = DECIDE.search(line)
    if not m:
        continue
    cols, filt, avail, served, fh = int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6])
    kind = "full hit" if fh else ("partial: some columns from cache" if served else
                                  ("probe found some, none used" if avail else "nothing cached"))
    dec[(kind, "filtered" if filt else "unfiltered")] += 1
print("== scan decisions")
for (kind, f), n in sorted(dec.items(), key=lambda kv: -kv[1]):
    print(f"  {kind:36} {f:10} {n:5}")

# ---- 3. scan time breakdown ---------------------------------------------------------------
print("== scan time breakdown (seconds summed over scans; scans overlap in time)")
runs = [(run, scan_times(run))] + ([(base, scan_times(base))] if base else [])
keys = ["total", "prep", "lookup", "lock_wait", "materialize", "read", "assemble", "populate",
        "postfilter", "make_batch"]
print("  " + "".join(f"{r:>14}" for r, _ in runs) + "   " + ("delta" if base else ""))
for k in keys:
    vals = [t.get(k, 0.0) for _, t in runs]
    delta = f"{vals[0] - vals[1]:+8.2f}" if base else ""
    print(f"  {k:12}" + "".join(f"{v:14.2f}" for v in vals) + "   " + delta)
print("  scans       " + "".join(f"{int(t['scans']):14d}" for _, t in runs))
