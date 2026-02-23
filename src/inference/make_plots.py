#!/usr/bin/env python3
"""
Plot metrics from ESM2 inference runs.

Reads all metrics_*.json files per output directory and aggregates across ranks.
Groups GPU utilization telemetry by hostname to show per-host averages.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_all_metrics(output_dir: Path):
    """Load all metrics_*.json files in a directory and aggregate.

    Returns:
        tuple: (timestamps_relative, tok_per_sec_summed, mean_tps, std_tps,
                num_gpus, tmin, tmax) or None if no files found.
    """
    metrics_files = sorted(output_dir.glob("metrics_*.json"))
    if not metrics_files:
        return None

    all_tok_per_sec = []
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

        gpu_stats = data.get("gpu_stats", {})
        num_gpus += len(gpu_stats)

        all_tok_per_sec.append(tps)
        print(f"  Loaded {mf.name}: {len(ts)} samples, {len(gpu_stats)} GPUs")

    # Sum tok/s across ranks at each time step.
    # Ranks may have different lengths; pad shorter ones with NaN.
    max_len = max(len(a) for a in all_tok_per_sec)
    padded = np.full((len(all_tok_per_sec), max_len), np.nan)
    for i, a in enumerate(all_tok_per_sec):
        padded[i, : len(a)] = a

    # Per-timestep sum across ranks (total throughput)
    tok_sum = np.nansum(padded, axis=0)
    mean_val = np.nanmean(tok_sum)
    std_val = np.nanstd(tok_sum)

    # Use first rank's timestamps for the time axis
    with open(metrics_files[0]) as f:
        data0 = json.load(f)
    ts_rel = np.array([e["timestamp"] for e in data0["timeseries"]]) - tmin

    return ts_rel, tok_sum[: len(ts_rel)], mean_val, std_val, num_gpus, tmin, tmax


def load_telemetry_by_host(telemetry_dir: Path, tmin: float, tmax: float):
    """Load telemetry files grouped by hostname.

    Returns:
        dict: {hostname: {"timestamps": array, "gpu_utils": {gpu_key: array}, "cpu_util": array}}
    """
    hosts = defaultdict(lambda: {"timestamps": [], "gpu_utils": defaultdict(list), "cpu_util": []})

    if not telemetry_dir.is_dir():
        return {}

    for filepath in sorted(telemetry_dir.glob("*.json")):
        with open(filepath) as f:
            data = json.load(f)

        hostname = data.get("checkpoint_metadata", {}).get("hostname", "unknown")

        for sample in data["metrics"]:
            ts = sample["timestamp"]
            if ts < tmin or ts > tmax:
                continue

            hosts[hostname]["timestamps"].append(ts)
            hosts[hostname]["cpu_util"].append(sample["metrics"].get("cpu_percent", 0))

            for key, val in sample["metrics"].items():
                if key.endswith("_utilization") and key.startswith("gpu_"):
                    hosts[hostname]["gpu_utils"][key].append(val)

    # Convert lists to arrays and drop GPUs with all-zero utilization
    for hostname in hosts:
        hosts[hostname]["timestamps"] = np.array(hosts[hostname]["timestamps"])
        hosts[hostname]["cpu_util"] = np.array(hosts[hostname]["cpu_util"])
        gpu_utils = {}
        for key in hosts[hostname]["gpu_utils"]:
            arr = np.array(hosts[hostname]["gpu_utils"][key])
            if np.any(arr > 0):
                gpu_utils[key] = arr
        hosts[hostname]["gpu_utils"] = gpu_utils

    return dict(hosts)


def plot_per_run(output_dir: Path, plots_dir: Path):
    """Generate per-run plots: GPU utilization by host + throughput.

    Returns:
        tuple: (num_gpus, mean_tps, std_tps) or None if no data.
    """
    print(f"\nReading data from {output_dir}")

    result = load_all_metrics(output_dir)
    if result is None:
        print(f"  No metrics files found in {output_dir}")
        return None

    ts_rel, tok_sum, mean_val, std_val, num_gpus, tmin, tmax = result
    print(f"  Total GPUs: {num_gpus}, throughput mean: {mean_val:.0f} +/- {std_val:.0f} tok/s")

    telemetry_dir = output_dir / "telemetry-results"
    hosts_data = load_telemetry_by_host(telemetry_dir, tmin, tmax)
    has_telemetry = bool(hosts_data)
    has_metrics = len(ts_rel) > 0

    if not has_metrics and not has_telemetry:
        print("  No data to plot")
        return num_gpus, mean_val, std_val

    num_hosts = max(len(hosts_data), 1)
    # One subplot per host (utilization) + one for throughput
    num_rows = (num_hosts if has_telemetry else 0) + (1 if has_metrics else 0)
    fig, axes = plt.subplots(num_rows, 1, figsize=(12, 3.5 * num_rows), sharex=True)
    if num_rows == 1:
        axes = [axes]

    ax_idx = 0

    # GPU/CPU utilization per host
    if has_telemetry:
        for hostname in sorted(hosts_data):
            hdata = hosts_data[hostname]
            ax = axes[ax_idx]
            ts_h = hdata["timestamps"] - tmin
            short_host = hostname.split(".")[0]

            # Plot individual GPU traces + compute per-host mean/std
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
                ax.fill_between(t_short, gpu_mean - gpu_std, gpu_mean + gpu_std,
                                color="black", alpha=0.15)
                host_avg = np.mean(gpu_mean)
                host_std = np.mean(gpu_std)
                print(f"  {short_host}: GPU util avg={host_avg:.1f}% +/- {host_std:.1f}%")

            ax.plot(ts_h, hdata["cpu_util"], color="tab:red", linestyle="--", label="CPU")
            ax.set_ylabel("Utilization (%)")
            ax.set_title(f"GPU/CPU Utilization — {short_host}")
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right", fontsize=8, ncol=4)
            ax_idx += 1

    # Throughput (summed across all ranks)
    if has_metrics:
        ax = axes[ax_idx]
        ax.plot(ts_rel, tok_sum[: len(ts_rel)], label="Tokens/sec (all ranks)", color="green")
        ax.axhline(mean_val, color="green", linestyle=":", alpha=0.5,
                    label=f"mean={mean_val:.0f}")
        ax.set_ylabel("Throughput (tok/s)")
        ax.set_xlabel("Time (seconds)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

    plt.tight_layout()
    output_file = plots_dir / f"plot_gpu_{num_gpus}.png"
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"  Saved plot to {output_file}")

    return num_gpus, mean_val, std_val


def plot_throughput_summary(tok_per_secs: dict, plots_dir: Path):
    """Plot throughput vs number of GPUs summary bar chart."""
    if not tok_per_secs:
        return

    gpus = sorted(tok_per_secs.keys())
    means = [tok_per_secs[n][0] for n in gpus]
    stds = [tok_per_secs[n][1] for n in gpus]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gpus, means, color=(0.25, 0.25, 0.25), marker="o", linestyle=":")
    ax.bar(gpus, means, yerr=stds, capsize=5, width=0.5)
    ax.set_title("Average Throughput vs. Number of GPUs (ESM2)")
    ax.set_ylabel("Average Throughput (tok/s)")
    ax.set_xlabel("Number of GPUs")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = plots_dir / "plot_tps.png"
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"\nSaved throughput summary to {output_file}")


def main(output_dirs: str):
    """
    Plot metrics from all inference runs under output_dirs.

    Args:
        output_dirs: Parent directory containing per-run output subdirectories
    """
    output_dirs = Path(output_dirs)
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    tok_per_secs = {}

    for output_dir in sorted(output_dirs.iterdir()):
        if not output_dir.is_dir():
            continue

        result = plot_per_run(output_dir, plots_dir)
        if result is not None:
            num_gpus, mean_val, std_val = result
            tok_per_secs[num_gpus] = (mean_val, std_val)

    plot_throughput_summary(tok_per_secs, plots_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ESM2 inference metrics")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Parent directory containing per-run output subdirectories (default: outputs)",
    )
    args = parser.parse_args()

    main(args.output_dir)
