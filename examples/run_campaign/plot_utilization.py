#!/usr/bin/env python3
import json
import glob
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def load_metrics(telemetry_dir):
    rows = []

    total_files = 0
    for file in sorted(telemetry_dir.glob("*.json")):
        total_files += 1
        with open(file) as f:
            data = json.load(f)

        hostname = data["checkpoint_metadata"]["hostname"]

        for entry in data["metrics"]:
            row = {
                "timestamp": entry["timestamp"],
                "hostname": hostname
            }

            metrics = entry["metrics"]

            # flatten metrics
            for k, v in metrics.items():
                row[k] = v

            rows.append(row)

    df = pd.DataFrame(rows)

    # convert timestamp to datetime
    print(df["timestamp"])
    df["time"] = df["timestamp"] - df["timestamp"].min()  #pd.to_datetime(df["timestamp"], unit="ms")
    print(f'read {total_files} files')

    return df


def main(telemetry_dir: str):
    """
    Plot CPU and GPU utilization metrics over time.

    Args:
        telemetry_dir: directory containing telemetry JSON files
    """

    telemetry_dir = Path(telemetry_dir)
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    df = load_metrics(telemetry_dir)

    # detect GPU utilization columns — only include GPUs with any non-zero activity
    all_gpu_cols = [c for c in df.columns if "gpu_" in c and "utilization" in c and not "memory"  in c]
    gpu_cols = [c for c in all_gpu_cols if df[c].max() > 0]
    print(gpu_cols)

    # remove rows where all GPUs are idle
    if gpu_cols:
        df = df[(df[gpu_cols] != 0).any(axis=1)]

    # compute averages
    if gpu_cols:
        df["gpu_avg_utilization"] = df[gpu_cols].mean(axis=1)

    # smoothing
    df["cpu_smooth"] = df["cpu_percent"].rolling(5).mean()
    if gpu_cols:
        df["gpu_avg_utilization_smooth"] = df["gpu_avg_utilization"].rolling(5).mean()

    # GPUs to plot
    gpus_to_plot = [
        int(col.split("_")[1]) for col in gpu_cols if col.endswith("utilization")
    ]

    num_plots = len(gpus_to_plot) + 2
    ncols = 2
    nrows = (num_plots + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), sharex=True)
    axes = axes.flatten()

    # ---- GPU plots ----
    for i, gpu in enumerate(gpus_to_plot):
        ax = axes[i]
        ax.plot(df["time"], df[f"gpu_{gpu}_utilization"], label=f"GPU {gpu}")
        ax.set_ylabel(f"GPU {gpu} Utilization (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

    # ---- CPU plot ----
    idx = len(gpus_to_plot)
    ax = axes[idx]
    ax.plot(df["time"], df["cpu_percent"], label="CPU Utilization")
    ax.set_ylabel("CPU Utilization (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # ---- Avg GPU plot ----
    ax = axes[idx + 1]
    if "gpu_avg_utilization" in df:
        ax.plot(df["time"], df["gpu_avg_utilization"], label="Avg GPU Utilization")
    ax.set_ylabel("Avg GPU Utilization (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    axes[idx + 1].set_xlabel("Time")

    plt.tight_layout()

    output_file = plots_dir / "utilization_plot.png"
    plt.savefig(output_file, dpi=150)

    plt.show()

    print(f"Plot saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ESM2 inference metrics")
    parser.add_argument(
        "--telemetry_dir",
        type=str,
        default="outputs",
        help="Parent directory containing per-run output subdirectories (default: outputs)",
    )
    args = parser.parse_args()

    main(args.telemetry_dir)

#python plot_utilization.py --telemetry_dir data/outputs_test/telemetry-results/