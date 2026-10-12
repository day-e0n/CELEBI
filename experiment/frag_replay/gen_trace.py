"""Turn the scan sequence into allocation traces for three cache granularities.

Trace lines:  A <entry_id> <n_buffers> <bytes>... <payload_bytes>   access one cache entry
              M                                                       measure (after each scan)
An entry is evicted as a unit. Buffers are separate allocations, the way cudf holds a
column (data, null mask, STRING offsets); a slab is one allocation of exactly PAGE bytes.
"""
import json, math
from pathlib import Path
F = str(Path(__file__).resolve().parent)
PAGE, ALIGN = 16 << 20, 256
d = json.load(open(f"{F}/sizes.json"))
rg_rows = {int(k): v for k, v in d["rg_rows"].items()}
cols, chars = d["cols"], d["chars"]
mask = lambda r: ((r + 511) // 512) * 64            # cudf::bitmask_allocation_size_bytes
al = lambda b: (b + ALIGN - 1) // ALIGN * ALIGN

def rg_parts(c, rg):
    """(rows, data_bytes) of column c in row group rg."""
    r = rg_rows[rg]
    return r, (chars[f"{c}|{rg}"] if cols[c]["string"] else r * cols[c]["width"])

def buffers(c, rows, data):
    b = [data, mask(rows)]
    if cols[c]["string"]: b.append((rows + 1) * 4)
    return [x for x in b if x > 0]

def split_rows(rows, rpp):
    out, r = [], 0
    while r < rows: out.append(min(rpp, rows - r)); r += rpp
    return out

def exact_pages(c, rg):     # the design before slabs: data cut at 16MiB, mask/offsets separate
    rows, data = rg_parts(c, rg)
    if cols[c]["string"]:
        rpp = max(1, math.ceil(PAGE / (data / rows + 4)))
    else:
        rpp = max(1, PAGE // cols[c]["width"])
    per_row = data / rows
    return [buffers(c, n, round(n * per_row)) for n in split_rows(rows, rpp)]

def slab_pages(c, rg):      # current: everything inside one PAGE allocation
    rows, data = rg_parts(c, rg)
    per_row = data / rows
    need = (lambda n: al(round(n * per_row)) + al((n + 1) * 4) + al(mask(n))) if cols[c]["string"] \
        else (lambda n: al(n * cols[c]["width"]) + al(mask(n)))
    lo, hi = 1, rows + 1                         # largest n with need(n) <= PAGE
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if need(mid) <= PAGE else (lo, mid)
    out = []
    for n in split_rows(rows, lo):
        payload = sum(buffers(c, n, round(n * per_row)))
        out.append(([PAGE], payload))
    return out

def emit(policy):
    ids, lines = {}, []
    eid = lambda k: ids.setdefault(k, len(ids))
    for s in d["scans"]:
        for c in s["cols"]:
            if not s["rgs"]: continue
            if policy == "batch":
                rows = sum(rg_rows[g] for g in s["rgs"])
                data = sum(rg_parts(c, g)[1] for g in s["rgs"])
                b = buffers(c, rows, data)
                lines.append(f"A {eid((c, tuple(s['rgs'])))} {len(b)} {' '.join(map(str, b))} {sum(b)}")
            else:
                for g in s["rgs"]:
                    if policy == "page":
                        for i, b in enumerate(exact_pages(c, g)):
                            lines.append(f"A {eid((c, g, i))} {len(b)} {' '.join(map(str, b))} {sum(b)}")
                    else:
                        for i, (b, payload) in enumerate(slab_pages(c, g)):
                            lines.append(f"A {eid((c, g, i))} 1 {b[0]} {payload}")
        lines.append("M")
    open(f"{F}/trace_{policy}.txt", "w").write("\n".join(lines) + "\n")
    sizes = [int(x) for l in lines if l[0] == "A" for x in l.split()[3:-1]]
    print(f"{policy:5}: {len(ids)} entries, {sum(1 for l in lines if l[0]=='A')} accesses, "
          f"alloc sizes {min(sizes)/2**20:.4f}-{max(sizes)/2**20:.1f} MiB, distinct sizes {len(set(sizes))}")

if __name__ == "__main__":
    for p in ("batch", "page", "slab"): emit(p)
