#!/usr/bin/env python3
"""Simulate the page cache as the engine actually implements it.

`cache_aware_query_reorder`'s model is a global LRU over (table, column) pairs,
scored by how many columns two adjacent queries share. Measured against SF100
runs, its predicted hit bytes correlate with the measured scan saving at
r = -0.16 -- it is not a weak predictor, it is the wrong quantity. Three
mismatches, all visible in the engine log:

  1. An entry is keyed by (file, FILTER), not by column. Two scans of the same
     table whose pushed-down predicates differ never share anything.
  2. Coverage is all-or-nothing: `try_assign_cached_entries` serves a scan only
     if the entry covers every column it reads. Sharing 3 of 4 columns is worth
     zero, not 75%.
  3. An entry stops widening at the per-entry admission ceiling (budget/2) and
     never shrinks back -- `admission_stop_widening` fires 480-732 times per
     run. Whichever query touches an entry first therefore decides which
     columns it holds for the whole workload.

(3) is what makes ORDER matter at all, and it is the opposite of what an
adjacency-overlap objective optimises: to make an entry useful you want the
WIDEST query to open it, and the most expensive queries to arrive after it is
wide.
"""

from __future__ import annotations

import json
from pathlib import Path

GIB = 1024 ** 3


class EntryCacheModel:
    def __init__(self, scan_requests: dict, column_bytes: dict,
                 budget_bytes: int = 6 * GIB, max_entry_bytes: int | None = None):
        self.requests = scan_requests
        self.column_bytes = column_bytes
        self.budget = budget_bytes
        # Mirrors fixed_page_admission_max_entry_bytes()'s budget_half default.
        self.cap = max_entry_bytes if max_entry_bytes is not None else budget_bytes // 2

    def col_bytes(self, table: str, column: str) -> int:
        return self.column_bytes.get(table, {}).get(column, 0)

    def request_bytes(self, request: dict) -> int:
        return sum(self.col_bytes(request["table"], c) for c in request["columns"])

    def requests_for(self, qnum) -> list[dict]:
        """One request per (query, table). Measured, not assumed: 22 solo runs
        showed the engine creating exactly 7 entries, one per table and all
        UNFILTERED -- `disable_filter_pushdown` is set whenever auto-caching is
        on, so the predicate never reaches the reader and never reaches the key
        either. The columns are the union of the scan's projections AND its
        filter columns (q6 projects 2 lineitem columns but caches 4), which
        reproduces the engine's column count on 26 of 29 (query, table) pairs.
        """

        per_table = self.requests.get(f"q{qnum}", {})
        return [{"table": t, "columns": c, "filter": ""} for t, c in per_table.items()]

    def run(self, sequence) -> dict:
        """Replay a query order; return per-query hit bytes and diagnostics."""

        entries: dict[tuple, dict] = {}
        lru: list[tuple] = []
        resident = 0
        per_query: dict[str, int] = {}
        demanded = 0
        frozen_events = 0

        def touch(key):
            if key in lru:
                lru.remove(key)
            lru.append(key)

        for qnum in sequence:
            key_q = f"q{qnum}"
            hit = 0
            for request in self.requests_for(qnum):
                table = request["table"]
                if table is None:
                    continue
                columns = set(request["columns"])
                need = self.request_bytes(request)
                demanded += need
                key = (table, request["filter"])
                entry = entries.get(key)

                if entry is not None and columns <= entry["columns"]:
                    hit += need                      # (2) all-or-nothing coverage
                    touch(key)
                    continue

                if entry is None:
                    entry = {"columns": set(), "bytes": 0}
                    entries[key] = entry
                missing = columns - entry["columns"]
                added = sum(self.col_bytes(table, c) for c in missing)
                if entry["bytes"] + added <= self.cap:   # (3) widen, or freeze
                    entry["columns"] |= missing
                    entry["bytes"] += added
                    resident += added
                else:
                    frozen_events += 1
                touch(key)

            per_query[key_q] = hit
            while resident > self.budget and len(lru) > 1:
                victim = lru.pop(0)
                resident -= entries[victim]["bytes"]
                del entries[victim]

        return {
            "per_query": per_query,
            "served": sum(per_query.values()),
            "demanded": demanded,
            "frozen_events": frozen_events,
            "entries": {f"{t}|{f[:24]}": e["bytes"] for (t, f), e in entries.items()},
        }


def load(scan_requests_path: str | Path, column_bytes_path: str | Path, **kwargs):
    return EntryCacheModel(json.loads(Path(scan_requests_path).read_text()),
                           json.loads(Path(column_bytes_path).read_text()), **kwargs)
