#!/usr/bin/env python3
# wdy start
"""Create a simple speedup summary and SVG chart from performance_test outputs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def query_matches(value: str, query: int) -> bool:
    normalized = value.strip().lower()
    return normalized == str(query) or normalized == f"q{query}"


def read_runtimes(path: Path, query: int) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not query_matches(row["query"], query):
                continue
            rows.append((int(row["iteration"]), float(row["runtime_s"])))
    rows.sort()
    return rows


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def write_summary(path: Path, one_gpu: list[tuple[int, float]], four_gpu: list[tuple[int, float]]):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "runtime_1gpu_s", "runtime_4gpu_s", "speedup_1gpu_over_4gpu"])
        for (it1, rt1), (it4, rt4) in zip(one_gpu, four_gpu):
            if it1 != it4:
                raise ValueError(f"iteration mismatch: 1GPU iter={it1}, 4GPU iter={it4}")
            writer.writerow([it1, rt1, rt4, rt1 / rt4 if rt4 else ""])

        one_warm = [rt for it, rt in one_gpu if it > 1]
        four_warm = [rt for it, rt in four_gpu if it > 1]
        one_med = median(one_warm or [rt for _, rt in one_gpu])
        four_med = median(four_warm or [rt for _, rt in four_gpu])
        writer.writerow([])
        writer.writerow(["metric", "runtime_1gpu_s", "runtime_4gpu_s", "speedup_1gpu_over_4gpu"])
        writer.writerow(["warm_median_excluding_iter0_iter1", one_med, four_med, one_med / four_med if four_med else ""])


def svg_line(points: list[tuple[float, float]], color: str) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>'


def write_svg(path: Path, one_gpu: list[tuple[int, float]], four_gpu: list[tuple[int, float]], title: str):
    width, height = 960, 540
    left, right, top, bottom = 80, 30, 60, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_values = [rt for _, rt in one_gpu] + [rt for _, rt in four_gpu]
    y_max = max(all_values) * 1.1 if all_values else 1.0
    iterations = [it for it, _ in one_gpu]
    x_min, x_max = min(iterations), max(iterations)
    x_span = max(1, x_max - x_min)

    def x_pos(iteration: int) -> float:
        return left + ((iteration - x_min) / x_span) * plot_w

    def y_pos(runtime: float) -> float:
        return top + plot_h - (runtime / y_max) * plot_h

    one_points = [(x_pos(it), y_pos(rt)) for it, rt in one_gpu]
    four_points = [(x_pos(it), y_pos(rt)) for it, rt in four_gpu]

    one_warm = [rt for it, rt in one_gpu if it > 1]
    four_warm = [rt for it, rt in four_gpu if it > 1]
    one_med = median(one_warm or [rt for _, rt in one_gpu])
    four_med = median(four_warm or [rt for _, rt in four_gpu])
    speedup = one_med / four_med if four_med else 0.0

    y_ticks = 5
    tick_lines = []
    for i in range(y_ticks + 1):
        value = y_max * i / y_ticks
        y = y_pos(value)
        tick_lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        tick_lines.append(
            f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" '
            f'font-size="12" fill="#374151">{value:.2f}</text>'
        )

    x_labels = []
    for it in iterations:
        x = x_pos(it)
        x_labels.append(
            f'<text x="{x:.1f}" y="{height-45}" text-anchor="middle" '
            f'font-size="12" fill="#374151">{it}</text>'
        )

    point_marks = []
    for x, y in one_points:
        point_marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb"/>')
    for x, y in four_points:
        point_marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#dc2626"/>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{left}" y="34" font-size="22" font-weight="700" fill="#111827">{title}</text>
  <text x="{left}" y="54" font-size="14" fill="#4b5563">Warm median speedup, excluding iter 0/1: {speedup:.2f}x</text>
  {''.join(tick_lines)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827" stroke-width="1.5"/>
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827" stroke-width="1.5"/>
  {svg_line(one_points, "#2563eb")}
  {svg_line(four_points, "#dc2626")}
  {''.join(point_marks)}
  {''.join(x_labels)}
  <text x="{width/2:.1f}" y="{height-15}" text-anchor="middle" font-size="14" fill="#111827">Iteration</text>
  <text x="22" y="{height/2:.1f}" text-anchor="middle" font-size="14" fill="#111827" transform="rotate(-90 22 {height/2:.1f})">Runtime (s)</text>
  <rect x="{width-255}" y="74" width="210" height="62" fill="white" stroke="#d1d5db"/>
  <line x1="{width-235}" y1="96" x2="{width-195}" y2="96" stroke="#2563eb" stroke-width="3"/>
  <text x="{width-185}" y="101" font-size="13" fill="#111827">1 GPU</text>
  <line x1="{width-235}" y1="120" x2="{width-195}" y2="120" stroke="#dc2626" stroke-width="3"/>
  <text x="{width-185}" y="125" font-size="13" fill="#111827">4 GPU</text>
</svg>
"""
    path.write_text(svg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-gpu-runtime", required=True, type=Path)
    parser.add_argument("--four-gpu-runtime", required=True, type=Path)
    parser.add_argument("--query", type=int, default=2)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--title", default="TPC-H Q2 1GPU vs 4GPU Runtime")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    one_gpu = read_runtimes(args.one_gpu_runtime, args.query)
    four_gpu = read_runtimes(args.four_gpu_runtime, args.query)
    if not one_gpu or not four_gpu:
        raise SystemExit("missing runtime rows for requested query")

    write_summary(args.out_dir / "q2_speedup_summary.csv", one_gpu, four_gpu)
    write_svg(args.out_dir / "q2_speedup.svg", one_gpu, four_gpu, args.title)
    print(f"wrote {args.out_dir / 'q2_speedup_summary.csv'}")
    print(f"wrote {args.out_dir / 'q2_speedup.svg'}")


if __name__ == "__main__":
    main()
# wdy end
