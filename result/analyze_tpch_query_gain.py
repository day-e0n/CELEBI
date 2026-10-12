#!/usr/bin/env python3
"""Per-query CELEBI gain on TPC-H SF50, with the query features to explain it.

Latency/operator work: result/tpch_sf50_operator_breakdown_samples.csv (A_th_base_r1..3 vs
A_th_best_r1..3, executions 2-3, 6 samples per query) -- the data behind the Figure 5
operator breakdown. Byte hit: result/cache_hit_rate_per_query.csv, condition
"paging + reorder" (HIT_th_ro_evict: same arrival order, same reordered order, same
future-aware eviction, separate trace-on run). Columns read: tpch_pin_columns.QUERY_COLUMNS,
sizes: experiment/column_bytes/tpch_sf50.json.

Clusters: 1-D k-means (k=3) on wall-time reduction.

  python3 result/analyze_tpch_query_gain.py   -> result/tpch_query_gain.csv
"""
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test" / "tpch_performance"))
from tpch_pin_columns import QUERY_COLUMNS  # noqa: E402

CELEBI_ORDER = [8, 5, 7, 15, 14, 6, 17, 20, 11, 2, 16, 9, 18, 22, 13, 3, 10, 4, 21, 12, 1, 19]
GIB = 2 ** 30


def kmeans_1d(values, k=3, rounds=100):
    ordered = sorted(values)
    centers = [ordered[int(i * (len(ordered) - 1) / (k - 1))] for i in range(k)]
    for _ in range(rounds):
        groups = [[] for _ in centers]
        for v in values:
            groups[min(range(k), key=lambda i: abs(v - centers[i]))].append(v)
        new = [statistics.mean(g) if g else c for g, c in zip(groups, centers)]
        if new == centers:
            break
        centers = new
    return centers


def main():
    samples = {}
    for r in csv.DictReader((ROOT / "result/tpch_sf50_operator_breakdown_samples.csv").open()):
        samples.setdefault((r["condition"], r["query"]), []).append(r)
    hits = {r["query"]: r for r in csv.DictReader((ROOT / "result/cache_hit_rate_per_query.csv").open())
            if r["workload"] == "TPC-H SF50" and r["condition"] == "paging + reorder"}
    sizes = json.loads((ROOT / "experiment/column_bytes/tpch_sf50.json").read_text())

    def cols(q):
        return {(t, c) for t, cs in QUERY_COLUMNS[q].items() for c in cs}

    def nbytes(columns):
        return sum(sizes[t][c] for t, c in columns)

    rows = []
    for q in range(1, 23):
        b, c = samples[("baseline", f"q{q}")], samples[("best", f"q{q}")]
        mean = lambda rs, k: statistics.mean(float(r[k]) for r in rs)  # noqa: E731
        wall_b, wall_c = mean(b, "wall_ms"), mean(c, "wall_ms")
        sd_b = statistics.stdev(float(r["wall_ms"]) for r in b)
        sd_c = statistics.stdev(float(r["wall_ms"]) for r in c)
        pos = CELEBI_ORDER.index(q)
        prev = CELEBI_ORDER[pos - 1] if pos else CELEBI_ORDER[-1]  # wraps across executions
        mine = cols(q)
        earlier3 = set().union(*(cols(p) for p in
                                 [CELEBI_ORDER[(pos - i) % 22] for i in (1, 2, 3)]))
        h = hits[f"q{q}"]
        rows.append({
            "query": f"q{q}",
            "base_wall_ms": round(wall_b, 1), "celebi_wall_ms": round(wall_c, 1),
            "wall_saved_ms": round(wall_b - wall_c, 1),
            "wall_reduction_pct": round(100 * (1 - wall_c / wall_b), 1),
            "wall_diff_over_sd": round((wall_b - wall_c) / ((sd_b ** 2 + sd_c ** 2) / 2) ** 0.5, 1),
            "work_reduction_pct": round(100 * (1 - mean(c, "total_ms") / mean(b, "total_ms")), 1),
            "scan_work_reduction_pct": round(100 * (1 - mean(c, "scan") / mean(b, "scan")), 1),
            "scan_share_of_base_work_pct": round(100 * mean(b, "scan") / mean(b, "total_ms"), 1),
            "join_share_of_base_work_pct": round(100 * mean(b, "join") / mean(b, "total_ms"), 1),
            "byte_hit_pct": float(h["byte_hit_pct"]),
            "requested_gib": float(h["total_gib"]),
            "tables": "+".join(sorted(QUERY_COLUMNS[q])),
            "n_tables": len(QUERY_COLUMNS[q]),
            "reads_lineitem": "lineitem" in QUERY_COLUMNS[q],
            "n_columns": len(mine),
            "column_gib": round(nbytes(mine) / GIB, 2),
            "lineitem_column_gib": round(nbytes({x for x in mine if x[0] == "lineitem"}) / GIB, 2),
            "celebi_position": pos + 1, "prev_query": f"q{prev}",
            "shared_with_prev_gib": round(nbytes(mine & cols(prev)) / GIB, 2),
            "shared_with_prev_pct": round(100 * nbytes(mine & cols(prev)) / nbytes(mine), 1),
            "shared_with_prev3_pct": round(100 * nbytes(mine & earlier3) / nbytes(mine), 1),
        })
    centers = sorted(kmeans_1d([r["wall_reduction_pct"] for r in rows]))
    names = ("low", "mid", "high")
    for r in rows:
        i = min(range(3), key=lambda j: abs(r["wall_reduction_pct"] - centers[j]))
        r["cluster"] = names[i]
    rows.sort(key=lambda r: -r["wall_reduction_pct"])
    out = ROOT / "result/tpch_query_gain.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("cluster centers (wall reduction %):", [round(c, 1) for c in centers])
    keys = ("query", "cluster", "base_wall_ms", "celebi_wall_ms", "wall_reduction_pct",
            "wall_diff_over_sd", "scan_work_reduction_pct", "scan_share_of_base_work_pct",
            "byte_hit_pct", "requested_gib", "tables", "column_gib", "celebi_position",
            "prev_query", "shared_with_prev_pct", "shared_with_prev3_pct")
    print("\t".join(keys))
    for r in rows:
        print("\t".join(str(r[k]) for k in keys))
    print("\ncluster means:")
    for name in names:
        g = [r for r in rows if r["cluster"] == name]
        stat = lambda k: round(statistics.mean(float(r[k]) for r in g), 1)  # noqa: E731
        print(name, len(g), {k: stat(k) for k in (
            "wall_reduction_pct", "base_wall_ms", "byte_hit_pct", "scan_share_of_base_work_pct",
            "join_share_of_base_work_pct", "requested_gib", "column_gib", "n_tables",
            "shared_with_prev_pct", "shared_with_prev3_pct")},
            "lineitem:", sum(r["reads_lineitem"] for r in g), "/", len(g))
    print(out)


if __name__ == "__main__":
    main()
