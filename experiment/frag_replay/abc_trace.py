"""Traces for the A -> B -> C experiment: q16, q17, q19 as recorded by a fresh Sirius run.

batch: one cache entry per data batch, holding every column of the batch, keyed by
       (columns, row groups) -- the granularity Sirius's data batches have.
page:  uniform 16 MiB pages keyed by (column, row group, page) -- the current design.
A measurement ("M") closes each query. Also writes abc_content.json: per query, which
(column, row group) bytes each design holds and how many copies.
"""
import json
from collections import Counter
from gen_trace import F, rg_rows, buffers, rg_parts, slab_pages

scans = json.load(open(f"{F}/abc_scans.json"))
content = {"batch": [], "page": []}
for policy in ("batch", "page"):
    ids, lines, copies, payload_of = {}, [], Counter(), {}
    eid = lambda k: ids.setdefault(k, len(ids))
    for q in "ABC":
        requested = 0
        for s in (x for x in scans if x["q"] == q):
            if policy == "batch":
                key = (tuple(s["cols"]), tuple(s["rgs"]))
                bufs = []
                for c in s["cols"]:
                    rows = sum(rg_rows[g] for g in s["rgs"])
                    bufs += buffers(c, rows, sum(rg_parts(c, g)[1] for g in s["rgs"]))
                lines.append(f"A {eid(key)} {len(bufs)} {' '.join(map(str, bufs))} {sum(bufs)}")
                requested += sum(bufs)
                if key not in payload_of:
                    payload_of[key] = True
                    for c in s["cols"]:
                        for g in s["rgs"]: copies[(c, g)] += 1
            else:
                for c in s["cols"]:
                    for g in s["rgs"]:
                        for i, (b, payload) in enumerate(slab_pages(c, g)):
                            lines.append(f"A {eid((c, g, i))} 1 {b[0]} {payload}")
                            requested += payload
                        copies[(c, g)] = 1
        lines.append("M")
        unique = sum(sum(buffers(c, rg_rows[g], rg_parts(c, g)[1])) for (c, g) in copies)
        held = sum(n * sum(buffers(c, rg_rows[g], rg_parts(c, g)[1])) for (c, g), n in copies.items())
        content[policy].append({"query": q, "requested": requested, "unique": unique,
                                "duplicate": held - unique})
    open(f"{F}/abc_trace_{policy}.txt", "w").write("\n".join(lines) + "\n")
json.dump(content, open(f"{F}/abc_content.json", "w"), indent=1)
for p, v in content.items():
    print(p, [(x["query"], round(x["unique"] / 2**30, 2), round(x["duplicate"] / 2**30, 2)) for x in v])
