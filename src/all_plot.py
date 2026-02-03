#!/usr/bin/env python3
"""
Plot metrics from ESM2 inference runs.

Visualizes GPU/CPU utilization and throughput over time.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(output_dirs: str, rank: int = 0):
    """
    Plot metrics from inference run.

    Args:
        output_dir: Directory containing metrics and telemetry results
        rank: Rank of the metrics file to read (default: 0)
    """

    tok_per_secs = {}
    output_dirs = Path(output_dirs)
    for output_dir in output_dirs.iterdir():
        tmin = None
        tmax = None

        print(f"Reading data from {output_dir}")

        metrics_filepath = os.path.join(output_dir, f"metrics_{rank}.json")

        try:
            with open(metrics_filepath) as f:
                log1 = json.load(f)

                ts1 = np.array([entry["timestamp"] for entry in log1["timeseries"]])
                if tmin is None or ts1.min() < tmin:
                    tmin = ts1.min()
                if tmax is None or ts1.max() > tmax:
                    tmax = ts1.max()
            tok_per_sec = np.array([m["tok_per_sec"] for m in log1["timeseries"]])
            ts1 = ts1 - tmin

            # Get number of GPUs from gpu_stats
            gpu_stats = log1.get("gpu_stats", {})
            num_gpu = len(gpu_stats)
            print(f"Metrics timeframe: {tmin:.2f} to {tmax:.2f}, {num_gpu} GPU(s)")
        except FileNotFoundError:
            print(f"Metrics file not found: {metrics_filepath}")
            tmin = 0
            tmax = 1e10
            num_gpu = 4  # fallback default

        mean_val = np.mean(tok_per_sec)
        std_val = np.std(tok_per_sec)
        tok_per_secs[num_gpu] = (mean_val, std_val)

        ts2_f = None
        cpu_util = None
        gpu_util_skeys = [f"gpu_{i}_utilization" for i in range(num_gpu)]
        gpu_utils = {}

        telemetry_dir = os.path.join(output_dir, "telemetry-results")

        print(telemetry_dir)

        if os.path.isdir(telemetry_dir):
            for filename in sorted(os.listdir(telemetry_dir)):
                if not filename.endswith(".json"):
                    continue

                filepath = os.path.join(telemetry_dir, filename)

                with open(filepath) as f:
                    log2 = json.load(f)

                ts2 = np.array([m["timestamp"] for m in log2["metrics"]])
                mask = (ts2 >= tmin) & (ts2 <= tmax)

                for gpu_util_skey in gpu_util_skeys:
                    gpu_data = np.array(
                        [m["metrics"].get(gpu_util_skey, 0) for m in log2["metrics"]]
                    )[mask]
                    if gpu_util_skey not in gpu_utils:
                        gpu_utils[gpu_util_skey] = gpu_data
                    else:
                        gpu_utils[gpu_util_skey] = np.concatenate(
                            [gpu_utils[gpu_util_skey], gpu_data]
                        )

                cpu_data = np.array([m["metrics"]["cpu_percent"] for m in log2["metrics"]])[mask]
                if cpu_util is None:
                    cpu_util = cpu_data
                else:
                    cpu_util = np.concatenate([cpu_util, cpu_data])

                if ts2_f is None:
                    ts2_f = ts2[mask]
                else:
                    ts2_f = np.concatenate([ts2_f, ts2[mask]])
        else:
            print(f"Telemetry directory not found: {telemetry_dir}")

        # Normalize timestamps
        if ts2_f is not None:
            ts2_f = ts2_f - tmin

        has_metrics = os.path.exists(metrics_filepath)
        has_telemetry = ts2_f is not None and len(ts2_f) > 0

        if has_metrics and has_telemetry:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        elif has_telemetry:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))
        elif has_metrics:
            fig, ax2 = plt.subplots(1, 1, figsize=(12, 4))
        else:
            print("No data to plot")
            return

        # GPU/CPU utilization
        if has_telemetry:
            for gpu_util_skey in gpu_util_skeys:
                if gpu_util_skey in gpu_utils and len(gpu_utils[gpu_util_skey]) > 0:
                    non_zero = np.count_nonzero(gpu_utils[gpu_util_skey])
                    print(
                        f"GPU {gpu_util_skey.split('_')[1]}: {non_zero} / {len(gpu_utils[gpu_util_skey])}"
                    )
                    # ax1.plot(ts2_f, gpu_utils[gpu_util_skey], label=f"GPU {gpu_util_skey.split('_')[1]}")
                    ax1.plot(ts2_f, gpu_utils[gpu_util_skey], label="GPU")
            if cpu_util is not None:
                ax1.plot(ts2_f, cpu_util, label="CPU")
            ax1.set_ylabel("Utilization (%)")
            ax1.set_title("Throughput and GPU/CPU Utilization Over Time (ESM2 model)")
            ax1.grid(True)
            ax1.legend(loc="best")

        # Tokens per second
        if has_metrics:
            ax2.plot(ts1, tok_per_sec, label="Tokens / Second", color="green")
            ax2.set_ylabel("Throughput")
            ax2.set_xlabel("Time (seconds)")
            ax2.grid(True)
            ax2.legend(loc="lower right")

        plt.tight_layout()
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        output_file = os.path.join(plots_dir, f"plot_gpu_{num_gpu}.png")
        plt.savefig(output_file)
        print(f"Saved plot to {output_file}")

    gpus = list(tok_per_secs.keys())
    gpus.sort()  # sorts the list
    means = [tok_per_secs[n][0] for n in gpus]
    stds = [tok_per_secs[n][1] for n in gpus]

    print(means)
    print(stds)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        gpus,
        means,
        color=(0.25, 0.25, 0.25),  # dim black
        marker="o",
        linestyle=":",
        #  label='Mean'
    )

    ax.bar(
        gpus,
        means,
        yerr=stds,  # Error bars
        capsize=5,  # Add caps to error bars
        width=0.5,
    )

    ax.set_title("Average Throughput vs. Number of GPUs (ESM2 model)")
    ax.set_ylabel("Avarage Throughput")
    ax.set_xlabel("Number of GPUs")
    ax.grid(True)
    plt.tight_layout()
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    output_file = os.path.join(plots_dir, "plot_tps.png")
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ESM2 inference metrics")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory containing metrics files (default: outputs)",
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="Rank of metrics file to read (default: 0)"
    )
    args = parser.parse_args()

    main(args.output_dir, args.rank)
