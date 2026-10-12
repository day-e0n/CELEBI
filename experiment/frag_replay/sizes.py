"""Decoded bytes per (column, row group) of ClickBench, plus the baseline scan sequence."""
import json, re, glob, duckdb
PQ = "/mnt/nvme/clickbench/hits_v2.parquet"
log = glob.glob("/home/dy1013/sirius_multi-gpu/experiment/SL_base/*/log_dir/*.log")[0]
rx = re.compile(r"parquet_materialize target_gpu=\d+ files=\S+ columns=(\S*) row_groups=(\S*)")
scans = []
for line in open(log, errors="replace"):
    m = rx.search(line)
    if m:
        cols = [c for c in m[1].split("|") if c]
        rgs = [int(x) for x in m[2].split(",") if x != ""]
        scans.append({"cols": cols, "rgs": rgs})
used = sorted({c for s in scans for c in s["cols"]})
types = dict(duckdb.sql(f"select column_name, column_type from (describe select * from read_parquet('{PQ}'))").fetchall())
W = {"BOOLEAN":1,"TINYINT":1,"UTINYINT":1,"SMALLINT":2,"USMALLINT":2,"INTEGER":4,"UINTEGER":4,"DATE":4,"FLOAT":4,
     "BIGINT":8,"UBIGINT":8,"DOUBLE":8,"TIMESTAMP":8,"TIMESTAMP_S":8,"TIMESTAMP_MS":8,"TIMESTAMP_NS":8}
rg_rows = dict(duckdb.sql(f"select row_group_id, any_value(row_group_num_rows) from parquet_metadata('{PQ}') group by 1").fetchall())
starts, acc = {}, 0
for rg in sorted(rg_rows): starts[rg] = acc; acc += rg_rows[rg]
strs = [c for c in used if types[c] in ("VARCHAR", "BLOB")]
case = " ".join(f"WHEN file_row_number < {starts[rg] + rg_rows[rg]} THEN {rg}" for rg in sorted(rg_rows))
chars = {}
if strs:
    sel = ", ".join(f'sum(strlen("{c}"))' for c in strs)
    q = f"select case {case} end as rg, {sel} from read_parquet('{PQ}', file_row_number=true) group by 1"
    for row in duckdb.sql(q).fetchall():
        for c, v in zip(strs, row[1:]): chars[f"{c}|{row[0]}"] = int(v or 0)
out = {"rg_rows": rg_rows, "cols": {}}
for c in used:
    t = types[c]
    out["cols"][c] = {"string": c in strs, "width": W.get(t, 0), "type": t}
out["chars"] = chars
out["scans"] = scans
json.dump(out, open("/tmp/claude-1002/-home-dy1013-sirius-multi-gpu/9134a118-eb26-4f12-844a-fccab0a76115/scratchpad/frag/sizes.json", "w"))
print(len(scans), "scans;", len(used), "columns used;", len(strs), "string;",
      "unknown widths:", [c for c in used if c not in strs and W.get(types[c], 0) == 0])
