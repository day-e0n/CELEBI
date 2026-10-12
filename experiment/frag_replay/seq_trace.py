"""Traces for a query sequence recorded by a fresh Sirius run (default: experiment/SEQ8_base).

Each query in the run becomes one step; a measurement ("M") closes it. Designs:
  batch  one entry per whole data batch, keyed by (columns, row groups)
  exact  pages of <=16 MiB of data, null mask and offsets allocated separately
  page   uniform 16 MiB pages
Writes seq_trace_<design>.txt and seq_meta.json (steps, bytes each step asks for, and for every
entry the (column, row group) content it holds, so duplicates can be counted later).

  python3 seq_trace.py [run_dir_name]
"""
import glob, json, re, sys
from gen_trace import F, rg_rows, buffers, rg_parts, slab_pages, exact_pages

run = sys.argv[1] if len(sys.argv) > 1 else "SEQ8_base"
log = glob.glob(f"{F}/../{run}/*/log_dir/*.log")[0]
rx = re.compile(r"parquet_materialize target_gpu=\d+ files=\S+ columns=(\S*) row_groups=(\S*)")
steps, cur = [], None
for line in open(log, errors="replace"):
    if "QueryBegin: SQL:" in line and line.split("SQL:", 1)[1].strip().upper().startswith("SELECT"):
        cur = {"sql": line.split("SQL:", 1)[1].strip(), "scans": []}
        steps.append(cur)
    m = rx.search(line)
    if m and cur is not None:
        cur["scans"].append({"cols": [c for c in m[1].split("|") if c],
                             "rgs": [int(x) for x in m[2].split(",") if x != ""]})

unit_bytes = lambda c, g: sum(buffers(c, rg_rows[g], rg_parts(c, g)[1]))
meta = {"steps": [{"sql": s["sql"][:90], "requested": 0} for s in steps], "entries": {}}
for design in ("batch", "exact", "page"):
    ids, lines, content = {}, [], {}
    def eid(key, units):
        if key not in ids:
            ids[key] = len(ids)
            content[ids[key]] = units
        return ids[key]
    for si, st in enumerate(steps):
        req = 0
        for s in st["scans"]:
            if not s["rgs"]:
                continue
            if design == "batch":
                bufs = []
                for c in s["cols"]:
                    rows = sum(rg_rows[g] for g in s["rgs"])
                    bufs += buffers(c, rows, sum(rg_parts(c, g)[1] for g in s["rgs"]))
                units = [[c, g, unit_bytes(c, g)] for c in s["cols"] for g in s["rgs"]]
                i = eid(("b", tuple(s["cols"]), tuple(s["rgs"])), units)
                lines.append(f"A {i} {len(bufs)} {' '.join(map(str, bufs))} {sum(bufs)}")
                req += sum(bufs)
            else:
                for c in s["cols"]:
                    for g in s["rgs"]:
                        pages = exact_pages(c, g) if design == "exact" else slab_pages(c, g)
                        for k, pg in enumerate(pages):
                            if design == "exact":
                                bufs, payload = pg, sum(pg)
                            else:
                                bufs, payload = pg[0], pg[1]
                            i = eid((c, g, k), [[f"{c}#{k}", g, payload]])
                            lines.append(f"A {i} {len(bufs)} {' '.join(map(str, bufs))} {payload}")
                            req += payload
        if design == "page":
            meta["steps"][si]["requested"] = req
        lines.append("M")
    open(f"{F}/seq_trace_{design}.txt", "w").write("\n".join(lines) + "\n")
    meta["entries"][design] = content
json.dump(meta, open(f"{F}/seq_meta.json", "w"))
for i, st in enumerate(meta["steps"]):
    print(i, round(st["requested"] / 2**30, 2), st["sql"][:60])
