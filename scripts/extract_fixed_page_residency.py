#!/usr/bin/env python3
"""Extract fixed-page cache residency from Sirius logs and plot it."""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path


LOG_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=('[^']*'|[^\s]+)")
TABLE_RE = re.compile(r"table='(.*)' fixed_cols=")


def parse_kv(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in KV_RE.findall(line):
        out[key] = value.strip("'")
    return out


def parse_table_key(line: str) -> str:
    match = TABLE_RE.search(line)
    return match.group(1) if match else ""


def timestamp_ms(line: str) -> float | None:
    match = LOG_TS_RE.match(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000.0
    except ValueError:
        return None


def query_num(query: str) -> int:
    return int(query[1:]) if query.startswith("q") else 10_000


def bytes_to_gib(value: float) -> float:
    return value / float(1024**3)


def iter_logs(run_root: Path) -> list[tuple[str, str, Path]]:
    logs: list[tuple[str, str, Path]] = []
    for condition_dir in sorted(run_root.iterdir()):
        if not condition_dir.is_dir() or condition_dir.name == "summary":
            continue
        for query_dir in sorted(condition_dir.glob("q*"), key=lambda p: query_num(p.name)):
            for log in sorted((query_dir / "log_dir").glob("sirius_*.log")):
                logs.append((condition_dir.name, query_dir.name, log))
    return logs


def extract_rows(run_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    events: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for condition, query, log_path in iter_logs(run_root):
        t0: float | None = None
        max_directory_bytes = 0
        last_directory_bytes = 0
        max_budget_after_by_device: dict[str, int] = {}
        last_budget_after_by_device: dict[str, int] = {}
        evicted_bytes = 0
        evicted_pages = 0
        directory_events = 0
        budget_events = 0
        resident_by_cache_key: dict[str, int] = {}
        max_total_directory_bytes = 0
        last_total_directory_bytes = 0

        for line in log_path.read_text(errors="replace").splitlines():
            ts = timestamp_ms(line)
            if ts is not None and t0 is None:
                t0 = ts
            rel_ms = (ts - t0) if ts is not None and t0 is not None else ""

            if "[fixed-page-cache] page_directory" in line:
                fields = parse_kv(line)
                cache_key = parse_table_key(line)
                resident_bytes = int(fields.get("resident_bytes", "0") or 0)
                resident_pages = int(fields.get("resident_pages", "0") or 0)
                directory_events += 1
                max_directory_bytes = max(max_directory_bytes, resident_bytes)
                last_directory_bytes = resident_bytes
                if cache_key:
                    resident_by_cache_key[cache_key] = resident_bytes
                    last_total_directory_bytes = sum(resident_by_cache_key.values())
                    max_total_directory_bytes = max(max_total_directory_bytes, last_total_directory_bytes)
                events.append(
                    {
                        "condition": condition,
                        "query": query,
                        "event": "page_directory",
                        "cache_key": cache_key,
                        "relative_ms": rel_ms,
                        "device": "",
                        "resident_bytes": resident_bytes,
                        "resident_gib": bytes_to_gib(resident_bytes),
                        "total_resident_bytes": last_total_directory_bytes,
                        "total_resident_gib": bytes_to_gib(last_total_directory_bytes),
                        "resident_pages": resident_pages,
                        "evicted_bytes": int(fields.get("evicted_bytes", "0") or 0),
                        "evicted_pages": int(fields.get("evicted_pages", "0") or 0),
                        "source_log": str(log_path),
                    }
                )

            elif "[fixed-page-cache] page_budget applied" in line:
                fields = parse_kv(line)
                device = fields.get("device", "")
                resident_after = int(fields.get("resident_bytes_after", "0") or 0)
                evicted_b = int(fields.get("evicted_bytes", "0") or 0)
                evicted_p = int(fields.get("evicted_pages", "0") or 0)
                budget_events += 1
                evicted_bytes += evicted_b
                evicted_pages += evicted_p
                if device:
                    max_budget_after_by_device[device] = max(
                        max_budget_after_by_device.get(device, 0), resident_after
                    )
                    last_budget_after_by_device[device] = resident_after
                events.append(
                    {
                        "condition": condition,
                        "query": query,
                        "event": "page_budget",
                        "relative_ms": rel_ms,
                        "device": device,
                        "resident_bytes": resident_after,
                        "resident_gib": bytes_to_gib(resident_after),
                        "resident_pages": "",
                        "evicted_bytes": evicted_b,
                        "evicted_pages": evicted_p,
                        "source_log": str(log_path),
                    }
                )

        if directory_events or budget_events:
            summaries.append(
                {
                    "condition": condition,
                    "query": query,
                    "directory_events": directory_events,
                    "budget_events": budget_events,
                    "max_cache_directory_resident_gib": bytes_to_gib(max_directory_bytes),
                    "last_cache_directory_resident_gib": bytes_to_gib(last_directory_bytes),
                    "max_cache_directory_total_resident_gib": bytes_to_gib(max_total_directory_bytes),
                    "last_cache_directory_total_resident_gib": bytes_to_gib(last_total_directory_bytes),
                    "max_budget_resident_total_gib": bytes_to_gib(sum(max_budget_after_by_device.values())),
                    "last_budget_resident_total_gib": bytes_to_gib(sum(last_budget_after_by_device.values())),
                    "evicted_gib": bytes_to_gib(evicted_bytes),
                    "evicted_pages": evicted_pages,
                    "source_log": str(log_path),
                }
            )

    return events, summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    paging = [row for row in rows if row["condition"] == "paging"]
    paging.sort(key=lambda row: query_num(str(row["query"])))
    queries = [str(row["query"]) for row in paging]
    cache = [float(row["max_cache_directory_resident_gib"]) for row in paging]
    budget = [float(row["max_budget_resident_total_gib"]) for row in paging]
    evicted = [float(row["evicted_gib"]) for row in paging]

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )

    x = list(range(len(queries)))
    width = 0.28
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar([i - width for i in x], cache, width=width, label="cache resident max", color="#4c78a8")
    ax.bar(x, budget, width=width, label="budget resident max", color="#f58518")
    ax.bar([i + width for i in x], evicted, width=width, label="evicted total", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(queries, rotation=40, ha="right")
    ax.set_ylabel("GiB")
    ax.set_title("Fixed-page cache residency from paging logs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    events, summary = extract_rows(args.run_root)
    write_csv(args.output_prefix.with_name(args.output_prefix.name + "_events.csv"), events)
    write_csv(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary)
    plot_summary(summary, args.output_prefix.with_name(args.output_prefix.name + "_summary.png"))
    print(args.output_prefix.with_name(args.output_prefix.name + "_events.csv"))
    print(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"))
    print(args.output_prefix.with_name(args.output_prefix.name + "_summary.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
