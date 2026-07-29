#!/usr/bin/env python3
"""Parse test_datasets/tpch-dbgen qgen output files (experiment/expB_qgen_streams/
stream_N.sql, one per RNG seed, each containing all 22 TPC-H queries in Q1..Q22
order) into a clean {(stream, qnum): sql_text} dict.

qgen emits each query as a `select ... ;` block, interleaved with stray
`where rownum <= N;` artifact lines (a `:n` row-limit substitution left over
from templates that use it) that are NOT part of the query and must be
stripped.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAMS_DIR = REPO_ROOT / "experiment" / "expB_qgen_streams"

ROWNUM_RE = re.compile(r"^\s*where rownum <= -?\d+;\s*$")
# qgen's default dialect emits `interval 'N' day (3)` (an Oracle-style
# day-precision qualifier on the interval literal) for Q1's date delta --
# DuckDB doesn't accept the trailing "(3)"; strip it.
INTERVAL_PRECISION_RE = re.compile(r"(\binterval\s+'[-\d]+'\s+(?:day|month|year))\s*\(\d+\)", re.IGNORECASE)


def parse_stream_file(path: Path) -> list[str]:
    """Returns 22 query SQL strings, in Q1..Q22 order.

    Only a `select` line at paren-depth 0 starts a new top-level query --
    several templates (Q2, Q11, Q15, Q17, Q20, Q22, ...) have a nested
    `(select ...)` subquery whose own `select` line must NOT be treated as a
    query boundary.
    """
    lines = path.read_text().splitlines()
    select_starts = []
    depth = 0
    # Q15's `create view revenue0 (...) as select ...; select ...; drop view
    # revenue0;` is three statements that belong to one logical query block --
    # suppress the next two would-be boundaries once we enter one.
    suppress_next_boundaries = 0
    for i, l in enumerate(lines):
        stripped = l.strip()
        if depth == 0 and stripped.lower().startswith("create view"):
            select_starts.append(i)
            suppress_next_boundaries = 2
        elif stripped == "select" and depth == 0:
            if suppress_next_boundaries > 0:
                suppress_next_boundaries -= 1
            else:
                select_starts.append(i)
        depth += l.count("(") - l.count(")")
    blocks = []
    for idx, start in enumerate(select_starts):
        end = select_starts[idx + 1] if idx + 1 < len(select_starts) else len(lines)
        block_lines = [l for l in lines[start:end] if not ROWNUM_RE.match(l)]
        text = "\n".join(block_lines).strip()
        text = text.rstrip(";").rstrip()
        text = INTERVAL_PRECISION_RE.sub(r"\1", text)
        blocks.append(text)
    return blocks


def load_all_streams(streams_dir: Path = STREAMS_DIR) -> dict[tuple[int, int], str]:
    out: dict[tuple[int, int], str] = {}
    for path in sorted(streams_dir.glob("stream_*.sql"), key=lambda p: int(p.stem.split("_")[1])):
        stream_id = int(path.stem.split("_")[1])
        blocks = parse_stream_file(path)
        if len(blocks) != 22:
            raise ValueError(f"{path}: expected 22 queries, got {len(blocks)}")
        for qnum, sql in enumerate(blocks, start=1):
            out[(stream_id, qnum)] = sql
    return out


def main() -> int:
    streams = load_all_streams()
    print(f"loaded {len(streams)} (stream, qnum) query instances")
    # sanity spot-checks
    for stream, qnum in [(1, 1), (1, 8), (5, 5)]:
        sql = streams[(stream, qnum)]
        print(f"--- stream {stream} q{qnum} ({len(sql)} chars) ---")
        print(sql[:200].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
