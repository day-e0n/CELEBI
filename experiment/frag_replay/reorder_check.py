"""Natural vs small-first-big-last order, same per-query batches (from experiment/SL_base,
execution 1), three executions each. Writes rc_<order>_<design>.txt traces."""
import glob, re, sys
from gen_trace import F, rg_rows, buffers, rg_parts, slab_pages, exact_pages

NATURAL = [q for q in range(1, 44) if q not in (28, 29)]
REORDERED = [int(x) for x in sys.argv[1].split(",")]
log = glob.glob(f"{F}/../SL_base/*/log_dir/*.log")[0]
rx = re.compile(r"parquet_materialize target_gpu=\d+ files=\S+ columns=(\S*) row_groups=(\S*)")
steps, cur = [], None
for line in open(log, errors="replace"):
    if "QueryBegin: SQL:" in line and line.split("SQL:", 1)[1].strip().upper().startswith("SELECT"):
        cur = []; steps.append(cur)
    m = rx.search(line)
    if m and cur is not None:
        cur.append(([c for c in m[1].split("|") if c], [int(x) for x in m[2].split(",") if x != ""]))
assert len(steps) == 3 * len(NATURAL), len(steps)
by_q = {q: steps[i] for i, q in enumerate(NATURAL)}

def emit(order, design, name):
    ids, lines = {}, []
    eid = lambda k: ids.setdefault(k, len(ids))
    for q in order * 3:
        for cols, rgs in by_q[q]:
            if not rgs: continue
            for c in cols:
                if design == "batch":
                    b = buffers(c, sum(rg_rows[g] for g in rgs), sum(rg_parts(c, g)[1] for g in rgs))
                    lines.append(f"A {eid((c, tuple(rgs)))} {len(b)} {' '.join(map(str, b))} {sum(b)}")
                    continue
                for g in rgs:
                    pages = exact_pages(c, g) if design == "exact" else slab_pages(c, g)
                    for k, pg in enumerate(pages):
                        bufs, payload = (pg, sum(pg)) if design == "exact" else pg
                        lines.append(f"A {eid((c, g, k))} {len(bufs)} {' '.join(map(str, bufs))} {payload}")
        lines.append("M")
    open(f"{F}/rc_{name}_{design}.txt", "w").write("\n".join(lines) + "\n")
    req = sum(int(l.split()[-1]) for l in lines if l[0] == "A")
    print(name, design, f"requested {req / 2**30:.1f} GiB")

for name, order in (("natural", NATURAL), ("reordered", REORDERED)):
    for design in ("batch", "exact", "page"):
        emit(order, design, name)
