#!/usr/bin/env python3
"""
Plot replica execution timeline from a campaign SLURM log.

Usage:
    python plot_replicas.py slurm-17715157.out [--out timeline.png]
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml

    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

_TS_RE = re.compile(r"\x1b\[2m(\d{2}:\d{2}:\d{2}\.\d{3})\x1b\[0m")
_START_RE = re.compile(r"starting replica '(\w+)'")
_FINISH_RE = re.compile(r"Replica '(\w+)' finished")
_ERROR_RE = re.compile(r"Replica '(\w+)' raised")
_GROUP_RE = re.compile(
    r"Registered group '(\w+)': replicas=(\d+) priority=(\d+) "
    r"min=(\d+) max=(\d+) deps=\[([^\]]*)\] dep_threshold=(\d+) "
    r"resources=\(cpus=(\d+), gpus=(\d+)\)"
)
_GPU_ASSIGN_RE = re.compile(r"GPU assign: '(\w+)' \u2192 GPU\(s\) \[([^\]]*)\]")
# "resources: cpus=USED/TOTAL  gpus=USED/TOTAL" from scheduler log
_USAGE_RE = re.compile(r"resources: cpus=(\d+)/(\d+)\s+gpus=(\d+)/(\d+)")
# "available: cpus=AVAIL/TOTAL  gpus=AVAIL/TOTAL" from replica-finished log
_AVAIL_RE = re.compile(r"available: cpus=(\d+)/(\d+)\s+gpus=(\d+)/(\d+)")
_TOTAL_RES_RE = re.compile(r"Resource pool: total_cpus=(\d+)\s+total_gpus=(\d+)")
_READY_RE = re.compile(r"Group '(\w+)' signaled ready")

GROUP_COLORS = {
    "inference": "#4C72B0",
    "md": "#DD8452",
    "miniapps": "#55A868",
    "dummy": "#C44E52",
}
DEFAULT_COLOR = "#8172B2"

GROUP_ORDER = ["md", "miniapps", "inference", "dummy"]


def parse_log(path: str):
    """Parse SLURM log; return spans, group_meta, resource_timeline, gpu_assignments."""
    starts: dict[str, datetime] = {}
    spans = []
    group_meta = {}
    gpu_assignments = {}
    resource_timeline = []  # (elapsed_s, used_cpus, total_cpus, used_gpus, total_gpus)
    t0_dt = None
    total_cpus = total_gpus = 0

    _iso_re = re.compile(r"^(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2}")
    date_ref = "1970-01-01"
    with open(path) as fh:
        for raw in fh:
            if m := _iso_re.match(raw):
                date_ref = m.group(1)
                break

    with open(path) as fh:
        for raw in fh:
            # Group registration lines (no timestamp required)
            if m := _GROUP_RE.search(raw):
                name = m.group(1)
                deps_raw = m.group(6)
                deps = [
                    d.strip().strip("'\"") for d in deps_raw.split(",") if d.strip().strip("'\"")
                ]
                group_meta[name] = {
                    "replicas": int(m.group(2)),
                    "priority": int(m.group(3)),
                    "min": int(m.group(4)),
                    "max": int(m.group(5)),
                    "deps": deps,
                    "dep_threshold": int(m.group(7)),
                    "cpus": int(m.group(8)),
                    "gpus": int(m.group(9)),
                }

            if m := _TOTAL_RES_RE.search(raw):
                total_cpus = int(m.group(1))
                total_gpus = int(m.group(2))

            ts_m = _TS_RE.search(raw)
            if ts_m is None:
                continue
            dt = datetime.strptime(f"{date_ref} {ts_m.group(1)}", "%Y-%m-%d %H:%M:%S.%f")
            if t0_dt is None:
                t0_dt = dt

            if m := _GPU_ASSIGN_RE.search(raw):
                rid = m.group(1)
                gpu_str = m.group(2).strip()
                gpu_ids = [int(x) for x in gpu_str.split(",") if x.strip()] if gpu_str else []
                gpu_assignments[rid] = gpu_ids

            # Resource usage from scheduler summary line
            if m := _USAGE_RE.search(raw):
                uc, tc, ug, tg = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                elapsed = (dt - t0_dt).total_seconds() if t0_dt else 0
                resource_timeline.append((elapsed, uc, tc, ug, tg))

            # Resource available from replica-finished line (convert to used)
            elif m := _AVAIL_RE.search(raw):
                ac, tc, ag, tg = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                elapsed = (dt - t0_dt).total_seconds() if t0_dt else 0
                resource_timeline.append((elapsed, tc - ac, tc, tg - ag, tg))

            if m := _START_RE.search(raw):
                starts[m.group(1)] = dt
            elif m := _FINISH_RE.search(raw):
                rid = m.group(1)
                if rid in starts:
                    group = rid.rsplit("_", 1)[0]
                    spans.append((rid, group, starts.pop(rid), dt, True))
            elif m := _ERROR_RE.search(raw):
                rid = m.group(1)
                if rid in starts:
                    group = rid.rsplit("_", 1)[0]
                    spans.append((rid, group, starts.pop(rid), dt, False))

    for rid, start in starts.items():
        group = rid.rsplit("_", 1)[0]
        spans.append((rid, group, start, start, None))

    # Sort resource timeline and deduplicate by elapsed
    resource_timeline.sort(key=lambda x: x[0])

    t0 = min(s[2] for s in spans) if spans else t0_dt
    return spans, group_meta, resource_timeline, gpu_assignments, t0, (total_cpus, total_gpus)


def plot(spans, group_meta, resource_timeline, gpu_assignments, t0, total_resources, out_path):
    if not spans:
        print("No replica events found.", file=sys.stderr)
        return

    def sort_key(s):
        rid, group, *_ = s
        idx = int(rid.rsplit("_", 1)[-1])
        g_idx = GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER)
        return (g_idx, idx)

    spans.sort(key=sort_key)

    has_resources = len(resource_timeline) > 0
    total_cpus, total_gpus = total_resources

    n_rows = len(spans)
    gantt_h = max(6, n_rows * 0.38)

    fig = plt.figure(figsize=(20, gantt_h + (3 if has_resources else 0) + 1))

    if has_resources:
        gs = gridspec.GridSpec(
            2,
            2,
            height_ratios=[gantt_h, 2.5],
            width_ratios=[3, 1],
            hspace=0.2,
            wspace=0.18,
        )
        ax_gantt = fig.add_subplot(gs[0, 0])
        ax_info = fig.add_subplot(gs[0, 1])
        ax_res = fig.add_subplot(gs[1, 0])
        ax_leg = fig.add_subplot(gs[1, 1])
        ax_leg.axis("off")
    else:
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.18)
        ax_gantt = fig.add_subplot(gs[0, 0])
        ax_info = fig.add_subplot(gs[0, 1])
        ax_res = None

    # ── Gantt chart ──────────────────────────────────────────────────────────
    yticks, ylabels = [], []
    group_row_ranges = {}  # group -> (first_row, last_row)

    prev_group = None
    for row, (rid, group, start, end, ok) in enumerate(spans):
        t_start = (start - t0).total_seconds()
        t_end = (end - t0).total_seconds() if end != start else t_start + 0.5
        bar_w = t_end - t_start

        color = GROUP_COLORS.get(group, DEFAULT_COLOR)
        edgecolor = "red" if ok is False else "none"
        lw = 1.5 if ok is False else 0
        alpha = 0.45 if ok is None else 0.88

        # Draw separator line between groups
        if group != prev_group and prev_group is not None:
            ax_gantt.axhline(row - 0.5, color="grey", lw=0.6, alpha=0.5, linestyle="--")
        prev_group = group

        # Track row ranges per group
        if group not in group_row_ranges:
            group_row_ranges[group] = [row, row]
        else:
            group_row_ranges[group][1] = row

        # Alternating background per group
        g_idx = GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER)
        if g_idx % 2 == 0:
            ax_gantt.axhspan(row - 0.5, row + 0.5, color="grey", alpha=0.04, linewidth=0)

        ax_gantt.barh(
            row,
            bar_w,
            left=t_start,
            height=0.72,
            color=color,
            edgecolor=edgecolor,
            linewidth=lw,
            alpha=alpha,
        )

        # Annotate bar with GPU assignment or CPU count
        gpu_ids = gpu_assignments.get(rid, [])
        meta = group_meta.get(group, {})
        if gpu_ids:
            ann_txt = f"gpu:{','.join(str(g) for g in gpu_ids)}"
        elif meta.get("cpus", 0) > 0:
            ann_txt = f"{meta['cpus']} cpu(s)"
        else:
            ann_txt = ""

        if ann_txt and bar_w > 0.5:
            ax_gantt.text(
                t_start + bar_w / 2,
                row,
                ann_txt,
                ha="center",
                va="center",
                fontsize=5.5,
                color="white",
                fontweight="bold",
                clip_on=True,
            )

        yticks.append(row)
        ylabels.append(rid)

    # Group section labels on the right of the Gantt
    for group, (r0, r1) in group_row_ranges.items():
        meta = group_meta.get(group, {})
        mid = (r0 + r1) / 2
        pri = meta.get("priority", "?")
        cpus = meta.get("cpus", 0)
        gpus = meta.get("gpus", 0)
        info = f"priority={pri}\ncpu={cpus}  gpu={gpus}"
        ax_gantt.text(
            1.002,
            1.0 - (mid + 0.5) / n_rows,
            info,
            transform=ax_gantt.transAxes,
            va="center",
            ha="left",
            fontsize=6.5,
            color=GROUP_COLORS.get(group, DEFAULT_COLOR),
            fontweight="bold",
        )

    # Build a lookup: group -> sorted list of (row, start, end, ok) for its spans
    group_spans = {}
    for row, (_, group, start, end, ok) in enumerate(spans):
        group_spans.setdefault(group, []).append((row, start, end, ok))

    # Dependency arrows:
    #   tail  — midpoint of first dep-group replica bar (where signal_ready likely fires)
    #   head  — start of first dependent-group replica bar
    #   vline — trigger moment (start of first dependent replica)
    for group, _ in group_row_ranges.items():
        meta = group_meta.get(group, {})
        for dep_name in meta.get("deps", []):
            if dep_name not in group_spans or group not in group_spans:
                continue

            dep_first = group_spans[dep_name][0]  # (row, start, end, ok) of first dep replica
            grp_first = group_spans[group][
                0
            ]  # (row, start, end, ok) of first replica of this group

            dep_row, dep_start, dep_end, _ = dep_first
            grp_row, grp_start, _grp_end, _ = grp_first

            dep_bar_mid_t = (dep_start - t0).total_seconds()
            if dep_end != dep_start:
                dep_bar_mid_t += ((dep_end - dep_start).total_seconds()) / 2

            grp_bar_start_t = (grp_start - t0).total_seconds()

            ax_gantt.annotate(
                "",
                xy=(grp_bar_start_t, grp_row),  # head: start of first dependent replica
                xytext=(dep_bar_mid_t, dep_row),  # tail: mid of first dep replica bar
                arrowprops=dict(
                    arrowstyle="->",
                    color="#555555",
                    lw=1.3,
                    connectionstyle="arc3,rad=0.35",
                ),
                annotation_clip=False,
            )
            ax_gantt.axvline(grp_bar_start_t, color="#555555", lw=0.6, linestyle=":", alpha=0.5)

    ax_gantt.set_yticks(yticks)
    ax_gantt.set_yticklabels(ylabels, fontsize=7)
    ax_gantt.set_xlabel("Elapsed time (s)", fontsize=9)
    ax_gantt.set_title("Campaign Manager Timeline", fontweight="bold", fontsize=12)
    ax_gantt.invert_yaxis()
    ax_gantt.grid(axis="x", linestyle="--", alpha=0.35)

    legend_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    legend_patches += [
        mpatches.Patch(facecolor="white", edgecolor="red", linewidth=1.2, label="error"),
        mpatches.Patch(color="grey", alpha=0.45, label="still running"),
    ]
    ax_gantt.legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.8)

    # ── Group info + dependency table ────────────────────────────────────────
    ax_info.axis("off")
    ax_info.set_title("Campaign Manager Config", fontweight="bold", fontsize=9, pad=4)

    if group_meta:
        info_groups = [g for g in GROUP_ORDER if g in group_meta] + [
            g for g in group_meta if g not in GROUP_ORDER
        ]

        col_labels = [
            "Work\nflow",
            "Priority",
            "CPUs",
            "GPUs",
            "min/max\nreplicas",
            "Total\nreplicas",
            "Deps",
        ]
        rows_data, row_colors = [], []
        for gname in info_groups:
            m = group_meta[gname]
            deps = ", ".join(m.get("deps", [])) or "—"
            rows_data.append(
                [
                    gname,
                    str(m.get("priority", "?")),
                    str(m.get("cpus", 0)),
                    str(m.get("gpus", 0)),
                    f"{m.get('min', 0)}/{m.get('max', 0)}",
                    str(m.get("replicas", "?")),
                    deps,
                ]
            )
            c = GROUP_COLORS.get(gname, DEFAULT_COLOR)
            row_colors.append([c] + ["#f5f5f5"] * (len(col_labels) - 1))

        tbl = ax_info.table(
            cellText=rows_data,
            colLabels=col_labels,
            cellColours=row_colors,
            loc="upper center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1.0, 1.5)

        # Style header row
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#333333")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # Dependency chain text
        dep_chains = []
        for gname in info_groups:
            m = group_meta[gname]
            deps = m.get("deps", [])
            if deps:
                thr = m.get("dep_threshold", 1)
                dep_chains.append(f"  {', '.join(deps)} →(≥{thr}) {gname}")

        if dep_chains:
            dep_str = "Dependency graph:\n" + "\n".join(dep_chains)
            ax_info.text(
                0.5,
                0.2,
                dep_str,
                transform=ax_info.transAxes,
                va="bottom",
                ha="center",
                fontsize=8,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4ff", edgecolor="#aabbdd"),
            )

        # Scheduling logic note
        sched_note = (
            "Scheduler:\n"
            "  Pass 1: guarantee min replicas (by priority)\n"
            "Pass 2: fill up to max replicas (by priority)\n"
            "Blocked by: resource limits & unmet deps"
        )
        ax_info.text(
            0.5,
            0.5,
            sched_note,
            transform=ax_info.transAxes,
            va="bottom",
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffbe6", edgecolor="#ccaa00"),
        )

    # ── Resource utilization subplot ─────────────────────────────────────────
    if ax_res is not None and resource_timeline:
        times = [t for t, *_ in resource_timeline]
        used_gpus = [ug for _, _, _, ug, _ in resource_timeline]
        used_cpus = [uc for _, uc, *_ in resource_timeline]
        tot_gpus = [tg for _, _, _, _, tg in resource_timeline]
        tot_cpus = [tc for _, _, tc, *_ in resource_timeline]

        ax_res.step(times, used_gpus, where="post", color="#4C72B0", lw=1.8, label="GPUs used")
        ax_res.fill_between(times, used_gpus, step="post", color="#4C72B0", alpha=0.15)
        if any(t > 0 for t in tot_gpus):
            ax_res.step(
                times,
                tot_gpus,
                where="post",
                color="#4C72B0",
                lw=0.8,
                linestyle="--",
                alpha=0.55,
                label="GPU total",
            )

        ax_res.set_ylabel("GPUs in use", color="#4C72B0", fontsize=8)
        ax_res.tick_params(axis="y", labelcolor="#4C72B0", labelsize=7)
        ax_res.set_ylim(bottom=0)

        ax_cpu = ax_res.twinx()
        ax_cpu.step(times, used_cpus, where="post", color="#DD8452", lw=1.8, label="CPUs used")
        ax_cpu.fill_between(times, used_cpus, step="post", color="#DD8452", alpha=0.12)
        if any(t > 0 for t in tot_cpus):
            ax_cpu.step(
                times,
                tot_cpus,
                where="post",
                color="#DD8452",
                lw=0.8,
                linestyle="--",
                alpha=0.55,
                label="CPU total",
            )

        ax_cpu.set_ylabel("CPUs in use", color="#DD8452", fontsize=8)
        ax_cpu.tick_params(axis="y", labelcolor="#DD8452", labelsize=7)
        ax_cpu.set_ylim(bottom=0)

        ax_res.set_xlabel("Elapsed time (s)", fontsize=8)
        ax_res.set_title("Resource Utilization (GPU / CPU)", fontweight="bold", fontsize=9)
        ax_res.grid(axis="x", linestyle="--", alpha=0.35)

        lines1, lbl1 = ax_res.get_legend_handles_labels()
        lines2, lbl2 = ax_cpu.get_legend_handles_labels()
        ax_res.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7, loc="upper right", framealpha=0.8)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


def parse_config(path: str) -> dict:
    """Load group_meta from a campaign config.yaml (authoritative source)."""
    if not _HAVE_YAML:
        print("PyYAML not installed — falling back to log-parsed group metadata", file=sys.stderr)
        return {}
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    group_meta = {}
    for name, wf in cfg.get("workflows", {}).items():
        group_meta[name] = {
            "replicas": int(wf.get("replicas", 1)),
            "priority": int(wf.get("priority", 0)),
            "min": int(wf.get("min_replicas", 0)),
            "max": int(wf.get("max_replicas", 0)),
            "deps": list(wf.get("dependencies", [])),
            "dep_threshold": int(wf.get("dependency_threshold", 1)),
            "cpus": int(wf.get("required_cpus", 0)),
            "gpus": int(wf.get("required_gpus", 0)),
        }
    return group_meta


def _default_out(log_path: str) -> str:
    stem = Path(log_path).stem  # e.g. "slurm-17716610"
    m = re.search(r"(\d+)", stem)
    run_num = m.group(1) if m else stem
    return f"replica_timeline_{run_num}.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="SLURM output file")
    parser.add_argument(
        "--config",
        default=None,
        help="Campaign config.yaml (authoritative source for group metadata). "
        "Auto-detected as config.yaml next to the log file if not given.",
    )
    parser.add_argument(
        "--out", default=None, help="Output PNG (default: replica_timeline_<run>.png)"
    )
    args = parser.parse_args()

    if args.out is None:
        args.out = _default_out(args.log)

    # Auto-detect config.yaml next to the log file
    if args.config is None:
        candidate = Path(args.log).parent / "config.yaml"
        if candidate.exists():
            args.config = str(candidate)

    spans, group_meta_log, resource_timeline, gpu_assignments, t0, total_resources = parse_log(
        args.log
    )

    if args.config:
        group_meta = parse_config(args.config)
        print(f"Loaded group metadata from config: {args.config}")
    else:
        group_meta = group_meta_log
        print("No config.yaml found — using group metadata parsed from log")

    print(
        f"Parsed {len(spans)} replica spans, "
        f"{len(group_meta)} groups, "
        f"{len(resource_timeline)} resource events, "
        f"{len(gpu_assignments)} GPU assignments"
    )
    plot(spans, group_meta, resource_timeline, gpu_assignments, t0, total_resources, args.out)


if __name__ == "__main__":
    main()

# python plot_replicas.py slurm-17715157.out --out replica_timeline.png
