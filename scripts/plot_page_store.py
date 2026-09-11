#!/usr/bin/env python3
"""What replacing the cache entry with a flat page store bought, on TPC-H SF50.

Two panels, because the headline number and the reason for it are different facts:

  (a) total query time per condition -- the result
  (b) time against cache budget -- WHY it is the result. The store spends what it
      is given, monotonically. The entry scheme it replaced could not: entries held
      a median of 6 MB and one absent column voided the whole bundle, so a larger
      budget went unused.

Run after the fig_* runs exist:
  pixi run -e duckdb-python python scripts/plot_page_store.py
"""
from __future__ import annotations

import collections
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "experiment"
NQ = 22


def total_seconds(bucket: Path) -> float | None:
    """Warm-execution total for one run, in seconds. None when the run is short."""
    per: dict[str, float] = collections.defaultdict(float)
    seen: dict[str, set] = collections.defaultdict(set)
    for row in csv.DictReader(bucket.open()):
        per[row["execution"]] += float(row["total_ms"])
        seen[row["execution"]].add(row["query"])
    warm = [e for e in per if e != "1" and len(seen[e]) == NQ]
    return statistics.mean(per[e] for e in warm) / 1000.0 if warm else None


def condition(tag: str) -> tuple[float, float] | None:
    runs = []
    for rep in (1, 2, 3):
        bucket = ROOT / f"{tag}_{rep}" / "bucket.csv"
        if not bucket.exists():
            continue
        value = total_seconds(bucket)
        if value is not None:
            runs.append(value)
    if not runs:
        return None
    return statistics.mean(runs), (statistics.stdev(runs) if len(runs) > 1 else 0.0)


def main() -> int:
    panel_a = [
        ("baseline\n(no cache)", "fig_base", ps.NEUTRAL),
        ("page store\n2 GB", "fig_page2GB", ps.LIGHT),
        ("page store\n4 GB", "fig_page4GB", ps.DARK),
        ("page store\n6 GB", "fig_page6GB", ps.ACCENT),
    ]
    measured = [(label, condition(tag), style) for label, tag, style in panel_a]
    missing = [label for label, value, _ in measured if value is None]
    if missing:
        print(f"missing: {', '.join(m.replace(chr(10), ' ') for m in missing)}")
    measured = [(label, value, style) for label, value, style in measured if value]
    if not measured:
        raise SystemExit("no runs found under experiment/fig_*")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.6))

    xs = range(len(measured))
    ax_a.bar(xs, [v[0] for _, v, _ in measured],
             yerr=[v[1] for _, v, _ in measured],
             error_kw=ps.ERRBAR, zorder=3,
             **{k: [s[k] for _, _, s in measured] if k == "facecolor" else measured[0][2][k]
                for k in ("facecolor", "edgecolor", "linewidth")})
    baseline = measured[0][1][0]
    for x, (_, (mean, _), _) in zip(xs, measured):
        delta = "" if x == 0 else f"\n{100 * (mean - baseline) / baseline:+.1f}%"
        ax_a.text(x, mean, f"{mean:.1f}s{delta}", ha="center", va="bottom", fontsize=7)
    ax_a.set_xticks(list(xs))
    ax_a.set_xticklabels([label for label, _, _ in measured], fontsize=7.5)
    ax_a.set_ylim(0, max(v[0] for _, v, _ in measured) * 1.28)
    ps.finish(ax_a, "query time (s)")
    ax_a.set_title("(a) TPC-H SF50, adversarial order", fontsize=8.5, pad=4)

    budgets = [2, 4, 6]
    store = [condition(f"fig_page{b}GB") for b in budgets]
    entry = None
    if all(store):
        ax_b.errorbar(budgets, [s[0] for s in store], yerr=[s[1] for s in store],
                      marker="o", markersize=4, linewidth=1.2,
                      color=ps.ACCENT["facecolor"], zorder=3, **ps.ERRBAR)
    if entry:
        # One point, drawn as a line: the entry scheme was run at 6 GB and could not
        # use it, so its budget response is flat by construction, not by measurement.
        ax_b.axhline(entry[0], linestyle="--", linewidth=1.0,
                     color=ps.DARK["facecolor"], label="entry scheme, 6 GB", zorder=2)
    if all(store):
        for x, (mean, _) in zip(budgets, store):
            ax_b.annotate(f"{mean:.1f}s", (x, mean), textcoords="offset points",
                          xytext=(0, 7), ha="center", fontsize=7)
    ax_b.set_xticks(budgets)
    ax_b.set_xlabel("fixed-page cache budget (GB)", fontsize=8.5)
    ps.finish(ax_b, "query time (s)")
    ax_b.set_title("(b) response to cache budget", fontsize=8.5, pad=4)

    fig.tight_layout()
    ps.save(fig, "fig_page_store_sf50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
