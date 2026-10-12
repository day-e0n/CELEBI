#!/usr/bin/env python3
"""Render freshly measured query latency and byte cache hits with equal panel widths."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_requested_cache_hits import collect, write_csv
from analyze_full_query_latency_and_hits import hit_case_path

CONDITIONS = (
    ("baseline", "Baseline", "#4c7caf"),
    ("paging_lru", "Baseline+page caching", "#54a34b"),
    ("reorder_future", "Baseline+page caching+reordering (CELEBI)", "#f5b12b"),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finish(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.tick_params(length=3, width=0.7)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.7)


def prepare(root, cb_paging_hit_root=None, cb_future_hit_root=None):
    latency_root, hit_root = root / "latency_runs", root / "hit_runs"
    manifests = [
        json.loads((p / "manifest.json").read_text()) for p in (latency_root, hit_root)
    ]
    assert manifests[0]["extension_sha256"] == manifests[1]["extension_sha256"]
    assert manifests[0]["source_sha256"] == manifests[1]["source_sha256"]
    assert digest(manifests[0]["extension"]) == manifests[0]["extension_sha256"]
    for retry_root in (cb_paging_hit_root, cb_future_hit_root):
        if not retry_root:
            continue
        retry = json.loads((retry_root / "manifest.json").read_text())
        assert retry["extension_sha256"] == manifests[0]["extension_sha256"]
        assert retry["source_sha256"] == manifests[0]["source_sha256"]
        assert retry["workloads"]["cb"] == manifests[0]["workloads"]["cb"]
        retry_job = json.loads((retry_root.parent / "job_status.json").read_text())
        assert retry_job["status"] == "complete" and not retry_job["overlaps"]
    exit_policy = None
    if cb_future_hit_root:
        future_manifest = json.loads((cb_future_hit_root / "manifest.json").read_text())
        exit_policy = future_manifest.get("post_workload_exit_wrapper")
        if exit_policy:
            assert digest(exit_policy["path"]) == exit_policy["sha256"]
    state = json.loads((root / "job_status.json").read_text())
    assert not state["overlaps"], "Concurrent GPU/NVMe measurement detected"
    reuse_tpch = "th" not in manifests[0]["workloads"]
    prior_tpch = (
        Path(__file__).resolve().parents[1] / "experiment/FULL_CACHE_PIN_20261007/runs"
    )
    prior_manifest = (
        json.loads((prior_tpch / "manifest.json").read_text()) if reuse_tpch else None
    )
    hit_sources = root / "hit_sources"
    for workload in ("cb", "th"):
        for condition in ("paging_lru", "reorder_future"):
            source = (
                hit_case_path("th", condition)
                if reuse_tpch and workload == "th"
                else hit_root / f"hit/{workload}_r1/{condition}"
            )
            if cb_paging_hit_root and workload == "cb" and condition == "paging_lru":
                source = cb_paging_hit_root / "hit/cb_r1/paging_lru"
            if (
                cb_future_hit_root
                and workload == "cb"
                and condition == "reorder_future"
            ):
                source = cb_future_hit_root / "hit/cb_r1/reorder_future"
            link = hit_sources / f"hit/{workload}_r1/{condition}"
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() and link.resolve() != source.resolve():
                link.unlink()
            if not link.is_symlink():
                link.symlink_to(source.resolve(), target_is_directory=True)
            assert link.resolve() == source.resolve()
    summary, query_hits, path_hits, execution_hits, hit_audit = collect(
        "full", hit_sources
    )
    excluded_hit_cases = []
    for path in hit_root.glob("hit/*/*/metadata.json"):
        meta = json.loads(path.read_text())
        if meta.get("comparison_valid") is not True:
            excluded_hit_cases.append(
                dict(
                    case=str(path.parent),
                    status=meta.get("status"),
                    returncode=meta.get("returncode"),
                    status_counts=meta.get("status_counts"),
                    reason="Post-workload shutdown fault; replaced by a separate same-configuration complete run.",
                )
            )
    query_means, samples, totals, cases = [], [], {}, []
    for workload, title, n in (("cb", "ClickBench", 43), ("th", "TPC-H SF50", 22)):
        totals[title] = {}
        reused = reuse_tpch and workload == "th"
        active_latency_root = prior_tpch if reused else latency_root
        plan = (prior_manifest if reused else manifests[0])["workloads"][workload]
        for condition, label, _ in CONDITIONS:
            case = active_latency_root / f"latency/{workload}_r1/{condition}"
            meta = json.loads((case / "metadata.json").read_text())
            assert meta["status"] == "complete" and meta["comparison_valid"], case
            assert meta["environment"]["SIRIUS_CACHE_HIT_TRACE"] == "0"
            assert meta["future_mode"] == (
                "evict" if condition == "reorder_future" else "off"
            )
            expected_cache = "0" if condition == "baseline" else "1"
            for setting in (
                "SIRIUS_ENABLE_FIXED_PAGE_REUSE",
                "SIRIUS_FIXED_PAGE_AUTO_CACHE",
                "SIRIUS_FIXED_WIDTH_PAGE_CACHE_ENABLED",
                "SIRIUS_VARIABLE_PAGE_CACHE_ENABLED",
            ):
                assert meta["environment"][setting] == expected_cache
            if condition == "reorder_future":
                hint_calls = 0
                for log in (case / "log_dir").glob("*.log"):
                    for line in log.open():
                        if "QueryBegin: SQL: CALL sirius_set_cache_hints('" in line:
                            assert "sirius_set_cache_hints('evict;" in line
                            hint_calls += 1
                assert hint_calls == n * 3, (case, hint_calls)
            expected_order = (
                plan["reordered_order"]
                if condition == "reorder_future"
                else plan["arrival_order"]
            )
            assert meta["order"] == expected_order
            records = list(csv.DictReader((case / "queries.csv").open()))
            expected = {(q, ex) for q in range(1, n + 1) for ex in (1, 2, 3)}
            assert len(records) == len(expected)
            assert {(int(r["query"]), int(r["execution"])) for r in records} == expected
            assert all(
                r["status"] == "ok" and math.isfinite(float(r["wall_s"]))
                for r in records
            )
            cumulative = 0.0
            for q in range(1, n + 1):
                selected = [
                    r
                    for r in records
                    if int(r["query"]) == q and int(r["execution"]) in (2, 3)
                ]
                values = [float(r["wall_s"]) for r in selected]
                assert len(values) == 2
                wall = statistics.mean(values)
                cumulative += wall
                query_means.append(
                    dict(
                        workload=title,
                        condition=condition,
                        label=label,
                        query=f"q{q}",
                        query_wall_s=wall,
                        cumulative_query_wall_s=cumulative,
                        samples=2,
                        measurement_source=(
                            "previous_measurement"
                            if reused
                            else "remeasured_after_avg_fix"
                        ),
                    )
                )
                samples.extend(
                    dict(
                        workload=title,
                        condition=condition,
                        query=f"q{q}",
                        execution=int(r["execution"]),
                        query_wall_s=float(r["wall_s"]),
                    )
                    for r in selected
                )
            assert math.isclose(
                cumulative, meta["steady_state_workload_wall_s"], abs_tol=1e-8
            )
            totals[title][condition] = cumulative
            if condition != "baseline":
                hit_meta = json.loads(
                    (
                        hit_sources / f"hit/{workload}_r1/{condition}/metadata.json"
                    ).read_text()
                )
                for key in (
                    "order",
                    "executions",
                    "gpu_device",
                    "cache_budget_bytes",
                    "min_free_bytes_per_gpu",
                    "future_mode",
                    "future_horizon",
                    "late_materialization",
                    "prefetch",
                    "odirect",
                ):
                    assert hit_meta[key] == meta[key], (case, key)
            cases.append(
                dict(
                    case=str(case),
                    metadata=meta,
                    query_csv_sha256=digest(case / "queries.csv"),
                    measurement_source=(
                        "previous_measurement" if reused else "remeasured_after_avg_fix"
                    ),
                )
            )
    return (
        query_means,
        samples,
        summary,
        query_hits,
        path_hits,
        execution_hits,
        dict(
            extension_sha256=manifests[0]["extension_sha256"],
            cb_paging_hit_retry_manifest=(
                str(cb_paging_hit_root / "manifest.json")
                if cb_paging_hit_root
                else None
            ),
            cb_future_hit_retry_manifest=(
                str(cb_future_hit_root / "manifest.json")
                if cb_future_hit_root
                else None
            ),
            reused_tpch=bool(reuse_tpch),
            tpch_previous_manifest=(
                str(prior_tpch / "manifest.json") if reuse_tpch else None
            ),
            tpch_extension_sha256=(
                prior_manifest["extension_sha256"]
                if reuse_tpch
                else manifests[0]["extension_sha256"]
            ),
            latency_manifest=str(latency_root / "manifest.json"),
            hit_manifest=str(hit_root / "manifest.json"),
            query_counts={"ClickBench": 43, "TPC-H SF50": 22},
            measured_executions=[2, 3],
            warmup_execution=1,
            independent_runs_per_condition=1,
            query_metric="Per-query execute().fetchall() elapsed time, accumulated in query-number order.",
            cache_metric="value_bytes_v2",
            cache_ratio_definition="100 * sum(cache-served decoded value bytes) / sum(requested decoded value bytes). Repeated accesses included; offsets, masks and allocation padding excluded.",
            latency_and_hit_runs="Separate trace-off/trace-on runs with matching settings.",
            totals=totals,
            latency_cases=cases,
            hit_cases=hit_audit,
            overlaps=state["overlaps"],
            excluded_hit_cases=excluded_hit_cases,
            cache_trace_post_workload_exit_policy=exit_policy,
        ),
    )


def render(query_means, hit_summary, output):
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.bbox": None,
        }
    ):
        width, height = 20.5, 5.5
        panel_width, panel_height = 5.3, 3.5
        fig = plt.figure(figsize=(width, height))
        axes = [
            fig.add_axes(
                [x / width, 1.05 / height, panel_width / width, panel_height / height]
            )
            for x in (0.82, 7.5, 14.18)
        ]
        max_latency = max(r["cumulative_query_wall_s"] for r in query_means)
        tick_step = 10 if max_latency > 40 else 5
        ymax = math.ceil((max_latency * 1.08) / tick_step) * tick_step
        for ax, title, n in zip(axes[:2], ("ClickBench", "TPC-H SF50"), (43, 22)):
            ends = []
            for condition, label, color in CONDITIONS:
                rows = [
                    r
                    for r in query_means
                    if r["workload"] == title and r["condition"] == condition
                ]
                assert len(rows) == n
                x = list(range(1, n + 1))
                y = [r["cumulative_query_wall_s"] for r in rows]
                ax.plot(
                    x,
                    y,
                    color=color,
                    marker="o",
                    markersize=3.3,
                    linewidth=1.6,
                    label=label,
                )
                ends.append((y[-1], color))
            ax.set_xticks(x, [f"q{q}" for q in x], rotation=90, fontsize=8)
            ax.set_xlim(0.4, n + 0.6)
            ax.set_ylim(0, ymax)
            ax.set_yticks(range(0, ymax + 1, tick_step))
            ax.set_xlabel("ClickBench query" if n == 43 else "TPC-H query", fontsize=13)
            ax.set_title(title, fontsize=16, pad=12)
            if n == 43:
                ax.set_ylabel("Cumulative query latency (s)", fontsize=13)
            finish(ax)
            upper = float("inf")
            for value, color in sorted(ends, reverse=True):
                position = min(value, upper - 0.05 * ymax)
                upper = position
                ax.annotate(
                    f"{value:.1f}s",
                    xy=(n, value),
                    xytext=(1.025, position),
                    textcoords=ax.get_yaxis_transform(),
                    fontsize=12,
                    va="center",
                    annotation_clip=False,
                )

        ax = axes[2]
        hit_map = {
            (r["workload"], r["condition"]): r["byte_hit_pct"] for r in hit_summary
        }
        max_hit = max(hit_map.values())
        hit_tick_step = 10
        hit_ymax = min(
            100, max(40, math.ceil((max_hit + 10) / hit_tick_step) * hit_tick_step)
        )
        for workload, title, color, marker, offset in (
            ("th", "TPC-H SF50", "#803e26", "o", -12),
            ("cb", "ClickBench", "#3d6b66", "s", 10),
        ):
            ys = [hit_map[workload, c] for c in ("paging_lru", "reorder_future")]
            ax.plot(
                [0, 1],
                ys,
                color=color,
                marker=marker,
                linewidth=1.6,
                markersize=8 if marker == "o" else 6,
                markerfacecolor="white" if marker == "o" else color,
                markeredgewidth=1.4,
                label=title,
            )
            for x, value in enumerate(ys):
                dy = offset if value > 3 or offset > 0 else 10
                ax.annotate(
                    f"{value:.1f}%",
                    (x, value),
                    xytext=(0, dy),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if dy > 0 else "top",
                    color=color,
                    fontsize=12,
                )
        ax.set_xticks([0, 1], ["Page caching", "CELEBI"], fontsize=11)
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(0, hit_ymax)
        ax.set_yticks(range(0, hit_ymax + 1, hit_tick_step))
        ax.set_ylabel("Byte cache hit ratio (%)", fontsize=13)
        ax.set_xlabel("Cache policy", fontsize=13)
        ax.set_title("Cache hit ratio", fontsize=16, pad=12)
        finish(ax)
        hit_legend = ax.legend(
            loc="upper left",
            frameon=True,
            edgecolor="#777777",
            facecolor="white",
            fontsize=12,
            handlelength=2.5,
            borderpad=0.6,
        )
        hit_legend.get_frame().set_linewidth(0.7)

        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=True,
            edgecolor="#777777",
            facecolor="white",
            fontsize=12,
            handlelength=2.5,
            columnspacing=2,
            borderpad=0.6,
        )
        legend.get_frame().set_linewidth(0.7)
        fig.text(
            0.5,
            0.04,
            (
                "ClickBench remeasured: all 43 queries; TPC-H SF50: previous 22-query measurements. "
                if any(
                    r.get("measurement_source") == "previous_measurement"
                    for r in query_means
                )
                else "ClickBench: all 43 queries; TPC-H SF50: 22 queries. "
            )
            + "Mean of executions 2–3; execution 1 excluded.\n"
            + "CELEBI includes future-aware eviction. Latency: trace OFF; cache ratio: matching trace ON runs.",
            ha="center",
            va="center",
            fontsize=9,
            color="#555555",
        )
        fig.canvas.draw()
        sizes = [
            (a.get_position().width * width, a.get_position().height * height)
            for a in axes
        ]
        assert all(
            math.isclose(w, panel_width) and math.isclose(h, panel_height)
            for w, h in sizes
        )
        layout = dict(
            figure_inches=[width, height],
            panel_inner_inches=sizes,
            panels_have_equal_inner_dimensions=True,
            latency_panels_share_y_limits=[0, ymax],
            cache_hit_y_limits=[0, hit_ymax],
            cache_hit_y_tick_step=hit_tick_step,
            consistent_panel_frames_and_legend_style=True,
        )
        for extension in ("png", "svg", "pdf"):
            path = output / f"query_latency_and_cache_hit_avg_fix.{extension}"
            fig.savefig(path, dpi=220, facecolor="white")
            print(path, flush=True)
        plt.close(fig)
    return layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cb-paging-hit-root", type=Path)
    parser.add_argument("--cb-future-hit-root", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    means, samples, hits, query_hits, paths, executions, provenance = prepare(
        args.input_root.resolve(),
        args.cb_paging_hit_root.resolve() if args.cb_paging_hit_root else None,
        args.cb_future_hit_root.resolve() if args.cb_future_hit_root else None,
    )
    write_csv(out / "query_latency_means.csv", means)
    write_csv(out / "query_latency_samples.csv", samples)
    write_csv(out / "cache_hit_summary.csv", hits)
    write_csv(out / "cache_hit_per_query.csv", query_hits)
    write_csv(out / "cache_hit_per_path.csv", paths)
    write_csv(out / "cache_hit_per_execution.csv", executions)
    provenance["layout"] = render(means, hits, out)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance["totals"], indent=2), flush=True)


if __name__ == "__main__":
    main()
