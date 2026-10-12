#!/usr/bin/env python3
"""One byte-weighted cache-hit figure from completed full-workload CELEBI traces."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
from pathlib import Path

from plot_requested_cache_hits import parse_logs
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result"


def main():
    rows, cases = [], []
    conditions = [("cb", "ClickBench", 43, "#3d6b66"),
                  ("th", "TPC-H SF50", 22, "#803e26")]
    for workload, title, n, _ in conditions:
        case = ROOT / f"experiment/FULL_CACHE_PIN_20261007/runs/hit/{workload}_r1/reorder_future"
        metadata = json.loads((case / "metadata.json").read_text())
        assert metadata["status"] == "complete" and metadata["comparison_valid"], case
        assert metadata["future_mode"] == "evict" and metadata["future_horizon"] == "execution", case
        assert metadata["environment"]["SIRIUS_CACHE_HIT_TRACE"] == "1", case
        with (case / "queries.csv").open() as source:
            queries = list(csv.DictReader(source))
        assert len(queries) == 3 * n and {(int(r["query"]), int(r["execution"])) for r in queries} == {
            (q, ex) for q in range(1, n + 1) for ex in (1, 2, 3)}, case
        assert all(r["status"] == "ok" for r in queries), case
        logs = sorted((case / "log_dir").glob("*.log"))
        data, excluded = parse_logs(logs, set(range(1, n + 1)), "reorder_future", "value_bytes_v2")
        totals = collections.Counter()
        for counter in data.values():
            totals.update(counter)
        total, reader = totals["total_bytes"], totals["read_bytes"]
        assert total > 0 and 0 <= reader <= total
        ratio = 100 * (total - reader) / total
        rows.append(dict(workload=title, condition="reorder_future", query_count=n, executions="2,3",
                         total_decoded_value_bytes=total, reader_decoded_value_bytes=reader,
                         cache_served_decoded_value_bytes=total - reader, cache_hit_pct=ratio,
                         trace_batches=totals["trace_batches"], metric="value_bytes_v2"))
        hint_count = 0
        for log in logs:
            for line in log.read_text().splitlines():
                if "QueryBegin: SQL: CALL sirius_set_cache_hints('" in line:
                    assert "sirius_set_cache_hints('evict;" in line, (case, line)
                    hint_count += 1
        assert hint_count == n * 3, (case, hint_count)
        cases.append(dict(case=str(case), metadata=metadata, actual_future_eviction_calls=hint_count,
                          excluded_metrics=dict(excluded),
                          logs=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in logs]))
        print(json.dumps(rows[-1]), flush=True)

    prefix = OUT / "cache_hit_ratio_celebi_full"
    with prefix.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(str(prefix) + "_provenance.json").write_text(json.dumps(dict(
        definition="100 * sum(total_bytes-read_bytes) / sum(total_bytes), across all traced decoded-value materializations",
        denominator="Access-weighted traced decoded value bytes; repeated accesses included, not distinct dataset bytes.",
        trace_paths=["parquet", "cached_provider", "late_refetch"],
        warmup_execution=1, measured_executions=[2, 3], independent_runs=1, cases=cases), indent=2) + "\n")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.bbox": None}):
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        fig.subplots_adjust(left=0.15, right=0.96, bottom=0.23, top=0.77)
        ys = [r["cache_hit_pct"] for r in rows]
        ax.bar([0, 1], ys, width=0.52, color=[c[3] for c in conditions],
               edgecolor="#34413f", linewidth=0.65, zorder=3)
        for x, y in enumerate(ys):
            ax.text(x, y + 1.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=17,
                    fontweight="medium", color=conditions[x][3])
        ax.set_xticks([0, 1], [f"{title}\n{n} queries" for _, title, n, _ in conditions], fontsize=11)
        ax.tick_params(axis="x", length=0, pad=9)
        ax.tick_params(axis="y", length=3, width=0.55, labelsize=10)
        ax.set_xlim(-0.65, 1.65)
        ax.set_ylim(0, 50)
        ax.set_yticks(range(0, 51, 10))
        ax.set_ylabel("Cache hit ratio (% of bytes)", fontsize=11, labelpad=9)
        ax.grid(axis="y", color="#e5e7e7", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#707575")
            ax.spines[side].set_linewidth(0.65)
        fig.suptitle("Cache hit ratio", x=0.555, y=0.965, fontsize=16, fontweight="medium")
        fig.text(0.555, 0.865, "CELEBI · page caching + reordering + future-aware eviction",
                 ha="center", fontsize=9.5, color="#4d5555")
        fig.text(0.555, 0.052, "Cache-served decoded value bytes / total traced decoded value bytes\n"
                 "All queries · executions 2–3 · warmup excluded", ha="center", fontsize=8.2,
                 color="#626a6a", linespacing=1.6)
        for ext in ("png", "svg", "pdf"):
            path = prefix.with_suffix(f".{ext}")
            fig.savefig(path, dpi=300, facecolor="white")
            print(path, flush=True)
        plt.close(fig)


if __name__ == "__main__":
    main()
