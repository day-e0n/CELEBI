#!/usr/bin/env python3
"""Validate every saved q28/q29 answer and summarize the isolated AVG experiment."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    expected = {
        r["query"]: sorted(r["results"], key=lambda x: x[0])
        for r in json.loads((root / "cpu_reference.json").read_text())
    }
    checks, means, phases = [], [], []
    for directory in (root / "compare", root / "compare_final"):
        for case in sorted(directory.iterdir()):
            if not case.is_dir():
                continue
            meta = json.loads((case / "metadata.json").read_text())
            assert meta["status"] == "complete", case
            for result in json.loads((case / "results.json").read_text()):
                q = result["query"]
                actual = sorted(result["results"], key=lambda x: x[0])
                assert len(actual) == len(expected[q]), (case, q)
                for cpu, gpu in zip(expected[q], actual):
                    assert len(cpu) == len(gpu)
                    for a, b in zip(cpu, gpu):
                        matches = (
                            math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-6)
                            if isinstance(a, float)
                            else a == b
                        )
                        assert matches, (case, q, cpu, gpu)
                checks.append(
                    dict(
                        case=case.name,
                        query=q,
                        execution=result["execution"],
                        matches_cpu=True,
                    )
                )
            rows = list(csv.DictReader((case / "queries.csv").open()))
            for q in (28, 29):
                samples = [
                    float(r["wall_s"])
                    for r in rows
                    if int(r["query"]) == q
                    and (meta["executions"] == 1 or int(r["execution"]) > 1)
                ]
                means.append(
                    dict(
                        case=case.name,
                        query=q,
                        samples_s=samples,
                        mean_s=statistics.mean(samples),
                    )
                )
            query, totals = None, {}
            for log in sorted((case / "log_dir").glob("*.log")):
                for line in log.read_text().splitlines():
                    match = re.search(
                        r"sirius_set_query_label\('.*?_q(28|29)_exec", line
                    )
                    if match:
                        query = int(match[1])
                    match = re.search(
                        r"\[groupby-phase\].*?phase=(\w+) elapsed_us=(\d+)", line
                    )
                    if match:
                        assert query is not None
                        key = (query, match[1])
                        totals[key] = totals.get(key, 0) + int(match[2])
            phases.extend(
                dict(case=case.name, query=q, phase=phase, summed_s=us / 1e6)
                for (q, phase), us in sorted(totals.items())
            )
    report = dict(
        input="/mnt/nvme/clickbench/hits_v2.parquet",
        workload=[28, 29],
        warmup_excluded=True,
        independent_runs_per_condition=1,
        correctness_float_relative_tolerance=1e-8,
        correctness_float_absolute_tolerance=1e-6,
        checked_result_sets=len(checks),
        correctness=checks,
        query_latency=means,
        synchronized_phase_diagnostic=phases,
    )
    (root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
