#!/usr/bin/env python3
"""Dump the exact (query, qgen stream, SQL) sequence that
measure_reorder_scaling_qgen.py builds for a given batch size N, so the
sequence can be inspected by hand instead of only seen through aggregate
overlap_ratio/latency numbers.

tiled_qnum_sequence/assign_streams are prefix-stable in N, so dumping the
largest N covers every smaller N's sequence as a prefix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from measure_reorder_scaling_qgen import assign_streams, tiled_qnum_sequence  # noqa: E402
from parse_qgen_streams import load_all_streams  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--streams-dir", type=Path, default=None)
    parser.add_argument("--num-streams", type=int, default=10)
    args = parser.parse_args()

    streams = load_all_streams(args.streams_dir) if args.streams_dir else load_all_streams()
    qnums = tiled_qnum_sequence(args.n)
    strms = assign_streams(qnums, args.num_streams)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for pos, (q, s) in enumerate(zip(qnums, strms), start=1):
            f.write(f"-- position {pos:>4}  query Q{q}  qgen stream {s}\n")
            f.write(streams[(s, q)])
            f.write(";\n\n")

    print(f"wrote {args.out} ({len(qnums)} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
