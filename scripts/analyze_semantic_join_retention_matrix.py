#!/usr/bin/env python3
# wdy start
"""Build semantic JOIN retention matrices from TPC-H SQL and Sirius stage summaries.

The older byte-only matrix used min(previous JOIN output bytes, second JOIN input
bytes), which is only a capacity upper bound. This script adds a SQL-derived
JOIN graph check, so reuse is credited only when two queries share JOIN lineage.

For each query Qi we extract canonical JOIN edges from SQL predicates such as:
  l.l_orderkey = o.o_orderkey  ->  lineitem.l_orderkey=orders.o_orderkey

For each ordered pair Qi -> Qj:
  semantic_similarity = |edges(Qi) intersect edges(Qj)| / |edges(Qi) union edges(Qj)|
  semantic_reuse_gb   = previous JOIN output GB * semantic_similarity

This is still a static upper bound: it ignores exact row-level predicate overlap,
projection compatibility, join type, and runtime batch placement. But unlike the
byte-only matrix, unrelated joins such as q4 orders-lineitem and q14 lineitem-part
score zero.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

QUERIES = [f"q{i}" for i in range(1, 23)]
TABLES = {
    "customer",
    "orders",
    "lineitem",
    "part",
    "partsupp",
    "supplier",
    "nation",
    "region",
}
DEFAULT_ALIAS = {
    "c": "customer",
    "customer": "customer",
    "o": "orders",
    "orders": "orders",
    "l": "lineitem",
    "lineitem": "lineitem",
    "p": "part",
    "part": "part",
    "ps": "partsupp",
    "partsupp": "partsupp",
    "s": "supplier",
    "supplier": "supplier",
    "n": "nation",
    "nation": "nation",
    "r": "region",
    "region": "region",
}
TABLE_ALIAS_RE = re.compile(
    r"\b(customer|orders|lineitem|part|partsupp|supplier|nation|region)\s+(?:as\s+)?([a-z][a-z0-9_]*)\b",
    re.IGNORECASE,
)
REL_ALIAS_RE = re.compile(
    r"(?:\bfrom|\bjoin|,)\s+([a-z_][a-z0-9_]*)\s+(?:as\s+)?([a-z][a-z0-9_]*)\b",
    re.IGNORECASE,
)
COLUMN_EQ_RE = re.compile(
    r"\b([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)\s*=\s*([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)\b",
    re.IGNORECASE,
)
GPU_PROCESSING_RE = re.compile(r"call\s+gpu_processing\s*\(\s*\"(.*)\"\s*\)\s*;?\s*$", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-summary", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def to_int(value: str | None) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def read_join_bytes(path: Path) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    join_input = {q: 0 for q in QUERIES}
    join_output = {q: 0 for q in QUERIES}
    join_events = {q: 0 for q in QUERIES}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            query = row.get("query", "")
            if query not in join_input:
                continue
            if row.get("stage_kind") != "JOIN":
                continue
            join_input[query] += to_int(row.get("input_bytes"))
            join_output[query] += to_int(row.get("output_bytes"))
            join_events[query] += to_int(row.get("events"))
    return join_input, join_output, join_events


def unwrap_gpu_processing(text: str) -> str:
    text = text.strip()
    match = GPU_PROCESSING_RE.match(text)
    if match:
        text = match.group(1)
    return text.replace('\\"', '"')


def read_query_sql(query_dir: Path, query: str) -> str:
    path = query_dir / f"{query}.sql"
    if not path.exists():
        return ""
    return unwrap_gpu_processing(path.read_text(errors="replace")).lower()


def alias_map(sql: str) -> dict[str, str]:
    aliases = dict(DEFAULT_ALIAS)
    for relation, alias in REL_ALIAS_RE.findall(sql):
        relation_l = relation.lower()
        alias_l = alias.lower()
        aliases[alias_l] = relation_l if relation_l in TABLES else f"__derived__:{relation_l}"
    for table, alias in TABLE_ALIAS_RE.findall(sql):
        table_l = table.lower()
        alias_l = alias.lower()
        if table_l in TABLES:
            aliases[table_l] = table_l
            # Do not override an alias that the generic FROM/JOIN parser already
            # resolved to a derived relation, e.g. revenue_view r in q15.
            if not aliases.get(alias_l, "").startswith("__derived__:"):
                aliases[alias_l] = table_l
    return aliases


def canonical_edge(left_table: str, left_col: str, right_table: str, right_col: str) -> str:
    left = f"{left_table}.{left_col.lower()}"
    right = f"{right_table}.{right_col.lower()}"
    a, b = sorted([left, right])
    return f"{a}={b}"


def extract_join_edges(sql: str) -> set[str]:
    aliases = alias_map(sql)
    edges: set[str] = set()
    for a1, c1, a2, c2 in COLUMN_EQ_RE.findall(sql):
        t1 = aliases.get(a1.lower())
        t2 = aliases.get(a2.lower())
        if t1 is None or t2 is None:
            continue
        if t1.startswith("__derived__:") or t2.startswith("__derived__:"):
            continue
        if t1 == t2:
            continue
        edges.add(canonical_edge(t1, c1, t2, c2))
    return edges


def read_join_edges(query_dir: Path) -> dict[str, set[str]]:
    return {q: extract_join_edges(read_query_sql(query_dir, q)) for q in QUERIES}


def semantic_score(prev_edges: set[str], second_edges: set[str]) -> float | None:
    if not prev_edges or not second_edges:
        return None
    union = prev_edges | second_edges
    if not union:
        return None
    return len(prev_edges & second_edges) / len(union)


def previous_coverage(prev_edges: set[str], second_edges: set[str]) -> float | None:
    if not prev_edges or not second_edges:
        return None
    return len(prev_edges & second_edges) / len(prev_edges)


def second_coverage(prev_edges: set[str], second_edges: set[str]) -> float | None:
    if not prev_edges or not second_edges:
        return None
    return len(prev_edges & second_edges) / len(second_edges)


def pair_metrics(
    qi: str,
    qj: str,
    join_output: dict[str, int],
    edges: dict[str, set[str]],
) -> dict[str, object]:
    prev_edges = edges[qi]
    second_edges = edges[qj]
    score = None if qi == qj else semantic_score(prev_edges, second_edges)
    common = prev_edges & second_edges
    cost_gb = join_output[qi] / 1e9 if join_output[qi] else 0.0
    if score is None or cost_gb == 0.0:
        useful_cost_gb = ""
        reuse_gb = ""
        efficiency = ""
    elif score == 0.0:
        useful_cost_gb = ""
        reuse_gb = 0.0
        efficiency = 0.0
    else:
        useful_cost_gb = cost_gb
        reuse_gb = cost_gb * score
        efficiency = score
    return {
        "previous_query": qi,
        "second_query": qj,
        "semantic_similarity": "" if score is None else score,
        "semantic_retention_efficiency": efficiency,
        "semantic_reuse_gb": reuse_gb,
        "useful_join_retention_cost_gb": useful_cost_gb,
        "previous_join_output_gb": "" if cost_gb == 0.0 else cost_gb,
        "common_join_edges": " ".join(sorted(common)),
        "previous_join_edges": " ".join(sorted(prev_edges)),
        "second_join_edges": " ".join(sorted(second_edges)),
        "previous_edge_coverage": "" if qi == qj or previous_coverage(prev_edges, second_edges) is None else previous_coverage(prev_edges, second_edges),
        "second_edge_coverage": "" if qi == qj or second_coverage(prev_edges, second_edges) is None else second_coverage(prev_edges, second_edges),
    }


def as_float(value: object) -> float | None:
    if value == "" or value is None:
        return None
    return float(value)


def write_long(path: Path, join_input: dict[str, int], join_output: dict[str, int], join_events: dict[str, int], edges: dict[str, set[str]]) -> None:
    fieldnames = [
        "previous_query",
        "second_query",
        "semantic_similarity",
        "semantic_retention_efficiency",
        "semantic_reuse_gb",
        "useful_join_retention_cost_gb",
        "previous_join_output_gb",
        "previous_edge_coverage",
        "second_edge_coverage",
        "common_join_edges",
        "previous_join_edges",
        "second_join_edges",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qi in QUERIES:
            for qj in QUERIES:
                writer.writerow(pair_metrics(qi, qj, join_output, edges))


def write_matrix(path: Path, join_output: dict[str, int], edges: dict[str, set[str]], column: str) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["previous_query"] + QUERIES)
        for qi in QUERIES:
            row = [qi]
            for qj in QUERIES:
                value = pair_metrics(qi, qj, join_output, edges)[column]
                row.append("" if value == "" else f"{float(value):.9f}")
            writer.writerow(row)


def write_query_signatures(path: Path, join_input: dict[str, int], join_output: dict[str, int], join_events: dict[str, int], edges: dict[str, set[str]]) -> None:
    with path.open("w", newline="") as f:
        fieldnames = ["query", "join_events", "join_input_gb", "join_output_gb", "join_edge_count", "join_edges"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for q in QUERIES:
            writer.writerow(
                {
                    "query": q,
                    "join_events": join_events[q],
                    "join_input_gb": "" if join_input[q] == 0 else join_input[q] / 1e9,
                    "join_output_gb": "" if join_output[q] == 0 else join_output[q] / 1e9,
                    "join_edge_count": len(edges[q]),
                    "join_edges": " ".join(sorted(edges[q])),
                }
            )


def plot_heatmap(output_dir: Path, join_output: dict[str, int], edges: dict[str, set[str]], column: str, filename: str, title: str, cbar_label: str, cmap: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: could not import plotting libraries: {exc}")
        return

    values = []
    for qi in QUERIES:
        row = []
        for qj in QUERIES:
            value = as_float(pair_metrics(qi, qj, join_output, edges)[column])
            row.append(np.nan if value is None else value)
        values.append(row)
    arr = np.array(values, dtype=float)
    masked = np.ma.masked_invalid(arr)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(masked, cmap=cmap, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_xticks(range(len(QUERIES)))
    ax.set_yticks(range(len(QUERIES)))
    ax.set_xticklabels(QUERIES, rotation=90)
    ax.set_yticklabels(QUERIES)
    ax.set_xlabel("Second query (Qj)")
    ax.set_ylabel("Previous query (Qi)")
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if not math.isfinite(value):
                continue
            text = f"{value:.1f}" if column.endswith("gb") else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color="black")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def plot_heatmaps(output_dir: Path, join_output: dict[str, int], edges: dict[str, set[str]]) -> None:
    plot_heatmap(
        output_dir,
        join_output,
        edges,
        "semantic_similarity",
        "join_semantic_similarity_heatmap.png",
        "JOIN semantic similarity by SQL join graph",
        "Jaccard(shared join edges)",
        "viridis",
    )
    plot_heatmap(
        output_dir,
        join_output,
        edges,
        "semantic_reuse_gb",
        "join_semantic_reuse_gb_heatmap.png",
        "Semantic JOIN reuse upper bound",
        "GB",
        "YlOrRd",
    )
    plot_heatmap(
        output_dir,
        join_output,
        edges,
        "useful_join_retention_cost_gb",
        "join_semantic_useful_retention_cost_gb_heatmap.png",
        "JOIN retention cost only for semantically related pairs",
        "GB",
        "YlOrRd",
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    join_input, join_output, join_events = read_join_bytes(args.stage_summary)
    edges = read_join_edges(args.query_dir)

    write_query_signatures(args.output_dir / "join_semantic_query_signatures.csv", join_input, join_output, join_events, edges)
    write_long(args.output_dir / "join_semantic_pair_long.csv", join_input, join_output, join_events, edges)
    write_matrix(args.output_dir / "join_semantic_similarity_matrix.csv", join_output, edges, "semantic_similarity")
    write_matrix(args.output_dir / "join_semantic_reuse_gb_matrix.csv", join_output, edges, "semantic_reuse_gb")
    write_matrix(args.output_dir / "join_semantic_useful_retention_cost_gb_matrix.csv", join_output, edges, "useful_join_retention_cost_gb")
    if not args.no_plots:
        plot_heatmaps(args.output_dir, join_output, edges)

    related = 0
    nonzero = 0
    for qi in QUERIES:
        for qj in QUERIES:
            if qi == qj:
                continue
            score = semantic_score(edges[qi], edges[qj])
            if score is None:
                continue
            nonzero += 1
            if score > 0:
                related += 1
    print(f"semantic related pairs: {related}/{nonzero}")
    print(f"wrote: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
