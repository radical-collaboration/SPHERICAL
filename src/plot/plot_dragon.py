#!/usr/bin/env python3
"""
Plot Dragon telemetry and inference metrics.

Two modes:

  --telemetry-dir PATH
      Plot GPU/CPU utilization directly from a single Dragon telemetry directory
      (*.json files with checkpoint_metadata + metrics[]).  No inference metrics
      required.  Equivalent to the old plot_utilization.py usage.

  --output-dirs PATH
      Scan a parent directory for per-run output subdirectories.  Each subdir
      may contain metrics_*.json (inference throughput) and/or a
      telemetry-results/ subdirectory (Dragon telemetry).  Generates one plot
      per run and a summary throughput chart.  Equivalent to make_plots.py.

Both Dragon telemetry sources use the same JSON schema:
    { "checkpoint_metadata": { "hostname": "..." },
      "metrics": [ { "timestamp": ..., "metrics": { "cpu_percent": ...,
                                                     "gpu_0_utilization": ..., ... } } ] }

Usage:
    # standalone telemetry mode
    python plot_dragon.py --telemetry-dir outputs/telemetry-results

    # multi-run inference mode
    python plot_dragon.py --output-dirs outputs
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_telemetry_by_host(telemetry_dir: Path, tmin: float = None, tmax: float = None) -> dict:
    """Load Dragon telemetry files grouped by hostname.

    Returns:
        {hostname: {"timestamps": array, "gpu_utils": {col: array}, "cpu_util": array}}
    """
    hosts = defaultdict(
        lambda: {
            "timestamps": [],
            "gpu_utils": defaultdict(list),
            "gpu_mems": defaultdict(list),
            "cpu_util": [],
        }
    )

    if not telemetry_dir.is_dir():
        return {}

    for filepath in sorted(telemetry_dir.glob("*.json")):
        with open(filepath) as f:
            data = json.load(f)

        hostname = data.get("checkpoint_metadata", {}).get("hostname", "unknown")

        for sample in data["metrics"]:
            ts = sample["timestamp"]
            if tmin is not None and ts < tmin:
                continue
            if tmax is not None and ts > tmax:
                continue

            hosts[hostname]["timestamps"].append(ts)
            hosts[hostname]["cpu_util"].append(sample["metrics"].get("cpu_percent", 0))

            for key, val in sample["metrics"].items():
                if key.endswith("_utilization") and key.startswith("gpu_") and "memory" not in key:
                    hosts[hostname]["gpu_utils"][key].append(val)
                elif key.startswith("gpu_") and key.endswith("_memory_used_gb"):
                    hosts[hostname]["gpu_mems"][key].append(val)

    result = {}
    for hostname, hdata in hosts.items():
        timestamps = np.array(hdata["timestamps"])
        cpu_util = np.array(hdata["cpu_util"])
        gpu_utils = {}
        for key, vals in hdata["gpu_utils"].items():
            arr = np.array(vals)
            if np.any(arr > 0):
                gpu_utils[key] = arr
        gpu_mems = {key: np.array(vals) for key, vals in hdata["gpu_mems"].items()}
        result[hostname] = {
            "timestamps": timestamps,
            "cpu_util": cpu_util,
            "gpu_utils": gpu_utils,
            "gpu_mems": gpu_mems,
        }

    return result


def load_inference_metrics(output_dir: Path):
    """Load metrics_*.json inference timeseries from an output directory.

    Returns:
        (ts_rel, tok_sum, mean_tps, std_tps, num_gpus, tmin, tmax) or None
    """
    metrics_files = sorted(output_dir.glob("metrics_*.json"))
    if not metrics_files:
        return None

    all_tps = []
    tmin = float("inf")
    tmax = float("-inf")
    num_gpus = 0

    for mf in metrics_files:
        with open(mf) as f:
            data = json.load(f)
        ts = np.array([e["timestamp"] for e in data["timeseries"]])
        tps = np.array([e["tok_per_sec"] for e in data["timeseries"]])
        tmin = min(tmin, ts.min())
        tmax = max(tmax, ts.max())
        num_gpus += len(data.get("gpu_stats", {}))
        all_tps.append(tps)
        print(f"  Loaded {mf.name}: {len(ts)} samples, {len(data.get('gpu_stats', {}))} GPUs")

    max_len = max(len(a) for a in all_tps)
    padded = np.full((len(all_tps), max_len), np.nan)
    for i, a in enumerate(all_tps):
        padded[i, : len(a)] = a
    tok_sum = np.nansum(padded, axis=0)

    with open(metrics_files[0]) as f:
        ts_rel = np.array([e["timestamp"] for e in json.load(f)["timeseries"]]) - tmin

    return (
        ts_rel,
        tok_sum[: len(ts_rel)],
        np.nanmean(tok_sum),
        np.nanstd(tok_sum),
        num_gpus,
        tmin,
        tmax,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_hosts(axes, ax_idx: int, hosts_data: dict, tmin: float) -> int:
    """Plot per-host GPU/CPU utilization subplots. Returns next ax_idx."""
    for hostname in sorted(hosts_data):
        hdata = hosts_data[hostname]
        ax = axes[ax_idx]
        ts_h = hdata["timestamps"] - tmin
        short_host = hostname.split(".")[0]

        all_gpu_vals = []
        for gpu_key in sorted(hdata["gpu_utils"]):
            vals = hdata["gpu_utils"][gpu_key]
            gpu_idx = gpu_key.split("_")[1]
            ax.plot(ts_h[: len(vals)], vals, alpha=0.4, label=f"GPU {gpu_idx}")
            all_gpu_vals.append(vals)

        if all_gpu_vals:
            min_len = min(len(v) for v in all_gpu_vals)
            stacked = np.array([v[:min_len] for v in all_gpu_vals])
            gpu_mean = np.mean(stacked, axis=0)
            gpu_std = np.std(stacked, axis=0)
            t_short = ts_h[:min_len]
            ax.plot(t_short, gpu_mean, color="black", linewidth=2, label="GPU avg")
            ax.fill_between(
                t_short, gpu_mean - gpu_std, gpu_mean + gpu_std, color="black", alpha=0.15
            )

        ax.plot(ts_h, hdata["cpu_util"], color="tab:red", linestyle="--", label="CPU")

        ax.set_ylabel("Utilization (%)")
        ax.set_title(f"GPU/CPU Utilization — {short_host}")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, ncol=4)
        ax_idx += 1

    return ax_idx


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------


def print_summary(hosts_data: dict) -> None:
    all_ts = np.concatenate([h["timestamps"] for h in hosts_data.values()])
    duration = all_ts.max() - all_ts.min()
    total_samples = sum(len(h["timestamps"]) for h in hosts_data.values())

    hdr = f"\n{'Node':<16} {'GPU':<6} {'Util mean':>10} {'Util max':>10} {'Mem mean (GB)':>14} {'Mem max (GB)':>13}"
    print(hdr)
    print("-" * len(hdr.rstrip()))
    for hostname in sorted(hosts_data):
        short = hostname.split(".")[0]
        gpu_utils = hosts_data[hostname]["gpu_utils"]
        gpu_mems = hosts_data[hostname]["gpu_mems"]
        first = True
        for gpu_key in sorted(gpu_utils):
            gpu_id = gpu_key.split("_")[1]
            mem_key = f"gpu_{gpu_id}_memory_used_gb"
            mem_vals = gpu_mems.get(mem_key, np.array([]))
            node_label = short if first else ""
            mem_mean = f"{np.mean(mem_vals):>14.2f}" if len(mem_vals) else f"{'—':>14}"
            mem_max = f"{np.max(mem_vals):>13.2f}" if len(mem_vals) else f"{'—':>13}"
            print(
                f"{node_label:<16} {'GPU ' + gpu_id:<6} "
                f"{np.mean(gpu_utils[gpu_key]):>10.1f} "
                f"{np.max(gpu_utils[gpu_key]):>10.1f} "
                f"{mem_mean} {mem_max}"
            )
            first = False

    print(f"\nDuration: {duration:.0f}s  ({duration / 60:.1f} min)  |  Samples: {total_samples}")


# ---------------------------------------------------------------------------
# Standalone telemetry mode  (replaces plot_utilization.py)
# ---------------------------------------------------------------------------


def plot_telemetry_standalone(telemetry_dir: Path, output: Path) -> None:
    """Plot GPU/CPU utilization from a single telemetry directory."""
    hosts_data = load_telemetry_by_host(telemetry_dir)
    if not hosts_data:
        print(f"No telemetry JSON files found in {telemetry_dir}")
        return

    all_ts = np.concatenate([h["timestamps"] for h in hosts_data.values()])
    tmin = all_ts.min()

    num_rows = len(hosts_data)
    fig, axes = plt.subplots(num_rows, 1, figsize=(12, 3.5 * num_rows), sharex=True)
    if num_rows == 1:
        axes = [axes]

    _plot_hosts(axes, 0, hosts_data, tmin)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close(fig)
    print_summary(hosts_data)
    print(f"Saved → {output}")


# ---------------------------------------------------------------------------
# Multi-run inference mode  (replaces make_plots.py)
# ---------------------------------------------------------------------------


def plot_per_run(output_dir: Path, plots_dir: Path):
    """Generate per-run plot: GPU utilization by host + throughput timeseries."""
    print(f"\nReading data from {output_dir}")

    result = load_inference_metrics(output_dir)
    telemetry_dir = output_dir / "telemetry-results"
    tmin = result[5] if result is not None else None
    tmax = result[6] if result is not None else None
    hosts_data = load_telemetry_by_host(telemetry_dir, tmin, tmax)

    has_telemetry = bool(hosts_data)
    has_metrics = result is not None and len(result[0]) > 0

    if not has_telemetry and not has_metrics:
        print("  No data to plot")
        return None

    if result is not None:
        ts_rel, tok_sum, mean_val, std_val, num_gpus, tmin, tmax = result
        print(f"  Total GPUs: {num_gpus}, throughput mean: {mean_val:.0f} +/- {std_val:.0f} tok/s")
    else:
        num_gpus, mean_val, std_val = 0, 0.0, 0.0

    num_rows = (len(hosts_data) if has_telemetry else 0) + (1 if has_metrics else 0)
    fig, axes = plt.subplots(num_rows, 1, figsize=(12, 3.5 * num_rows), sharex=True)
    if num_rows == 1:
        axes = [axes]

    ax_idx = 0
    if has_telemetry:
        ax_idx = _plot_hosts(axes, ax_idx, hosts_data, tmin)

    if has_metrics:
        ax = axes[ax_idx]
        ax.plot(ts_rel, tok_sum[: len(ts_rel)], label="Tokens/sec (all ranks)", color="green")
        ax.axhline(mean_val, color="green", linestyle=":", alpha=0.5, label=f"mean={mean_val:.0f}")
        ax.set_ylabel("Throughput (tok/s)")
        ax.set_xlabel("Time (seconds)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

    plt.tight_layout()
    output_file = plots_dir / f"plot_gpu_{num_gpus}.png"
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"  Saved → {output_file}")

    return num_gpus, mean_val, std_val


def plot_throughput_summary(tok_per_secs: dict, plots_dir: Path) -> None:
    """Bar chart: average throughput vs number of GPUs."""
    if not tok_per_secs:
        return
    gpus = sorted(tok_per_secs)
    means = [tok_per_secs[n][0] for n in gpus]
    stds = [tok_per_secs[n][1] for n in gpus]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gpus, means, color=(0.25, 0.25, 0.25), marker="o", linestyle=":")
    ax.bar(gpus, means, yerr=stds, capsize=5, width=0.5)
    ax.set_title("Average Throughput vs. Number of GPUs")
    ax.set_ylabel("Average Throughput (tok/s)")
    ax.set_xlabel("Number of GPUs")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = plots_dir / "plot_tps.png"
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"\nSaved throughput summary → {output_file}")


def main_output_dirs(output_dirs: Path, plots_dir: Path) -> None:
    tok_per_secs = {}
    for output_dir in sorted(output_dirs.iterdir()):
        if not output_dir.is_dir():
            continue
        result = plot_per_run(output_dir, plots_dir)
        if result is not None:
            num_gpus, mean_val, std_val = result
            tok_per_secs[num_gpus] = (mean_val, std_val)
    plot_throughput_summary(tok_per_secs, plots_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Dragon telemetry and inference metrics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--telemetry-dir",
        help="Single Dragon telemetry directory (standalone mode)",
    )
    group.add_argument(
        "--output-dirs",
        help="Parent directory of per-run output subdirectories (multi-run mode)",
    )
    parser.add_argument(
        "--output",
        default="utilization.png",
        help="Output file for standalone mode (default: utilization.png)",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        help="Directory to write plots into for multi-run mode (default: plots/)",
    )
    args = parser.parse_args()

    if args.telemetry_dir:
        plot_telemetry_standalone(Path(args.telemetry_dir), Path(args.output))
    else:
        plots_dir = Path(args.plots_dir)
        plots_dir.mkdir(exist_ok=True)
        main_output_dirs(Path(args.output_dirs), plots_dir)

# Standalone:   python src/plot/plot_dragon.py --telemetry-dir outputs/telemetry-results
# Multi-run:    python src/plot/plot_dragon.py --output-dirs outputs --plots-dir plots
