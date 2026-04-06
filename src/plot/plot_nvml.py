#!/usr/bin/env python3
"""
Plot GPU utilization and memory from nvml-telemetry checkpoints.

Reads nvml_checkpoint_*.json files written by NvmlMonitor.
One subplot per node; each subplot shows all GPUs on that node.

Usage:
    python plot_nvml.py [--telemetry-dir nvml-telemetry] [--output gpu_util.png]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_telemetry_by_host(telemetry_dir: str) -> dict:
    """Load nvml checkpoints grouped by hostname.

    Returns:
        {hostname: pd.DataFrame} with columns timestamp, time_s,
        gpu_N_utilization, gpu_N_memory_used_gb, ...
    """
    host_records = defaultdict(list)

    for path in sorted(Path(telemetry_dir).glob("nvml_checkpoint_*.json")):
        # filename: nvml_checkpoint_{hostname}_{ts}.json
        stem = path.stem  # nvml_checkpoint_{hostname}_{ts}
        parts = stem.split("_", 2)  # ["nvml", "checkpoint", "{hostname}_{ts}"]
        # hostname contains dots; split on last underscore to get timestamp
        tail = parts[2]  # e.g. "gpub054.delta.ncsa.illinois.edu_1775496342"
        last_us = tail.rfind("_")
        hostname = tail[:last_us]

        with open(path) as f:
            data = json.load(f)

        for sample in data["metrics"]:
            row = {"timestamp": sample["timestamp"],
                   "cpu_percent": sample["metrics"].get("cpu_percent", 0)}
            for key, val in sample["metrics"].items():
                if key.startswith("gpu_") and key.endswith("_utilization") and "memory" not in key:
                    row[key] = val
                elif key.startswith("gpu_") and key.endswith("_memory_used_gb"):
                    row[key] = val
            host_records[hostname].append(row)

    result = {}
    for hostname, records in host_records.items():
        df = pd.DataFrame(records).drop_duplicates("timestamp").sort_values("timestamp")
        df["time_s"] = df["timestamp"] - df["timestamp"].iloc[0]
        result[hostname] = df

    return result


def plot(hosts: dict, output: str) -> None:
    """One subplot per node: utilization on left y-axis, memory on right y-axis."""
    n_hosts = len(hosts)
    fig, axes = plt.subplots(n_hosts, 1, figsize=(14, 4 * n_hosts), sharex=True)
    if n_hosts == 1:
        axes = [axes]

    for ax, hostname in zip(axes, sorted(hosts)):
        df = hosts[hostname]
        short = hostname.split(".")[0]
        util_cols = sorted(c for c in df.columns if c.endswith("_utilization"))
        mem_cols  = sorted(c for c in df.columns if c.endswith("_memory_used_gb"))

        # Individual GPU util lines (faded) + bold average
        util_vals = []
        for col in util_cols:
            gpu_id = col.split("_")[1]
            ax.plot(df["time_s"], df[col], alpha=0.4, label=f"GPU {gpu_id}")
            util_vals.append(df[col].values)

        if util_vals:
            avg = np.mean(util_vals, axis=0)
            std = np.std(util_vals, axis=0)
            ax.plot(df["time_s"], avg, color="black", linewidth=2, label="GPU avg")
            ax.fill_between(df["time_s"], avg - std, avg + std,
                            color="black", alpha=0.15)

        ax.plot(df["time_s"], df["cpu_percent"], color="tab:red",
                linestyle="--", label="CPU")

        ax.set_ylabel("Utilization (%)")
        ax.set_ylim(-5, 105)
        ax.set_title(f"{short} — GPU/CPU Utilization")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, ncol=4)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved → {output}")


def print_summary(hosts: dict) -> None:
    all_ts = np.concatenate([df["timestamp"].values for df in hosts.values()])
    duration = all_ts.max() - all_ts.min()
    total_samples = sum(len(df) for df in hosts.values())

    hdr = f"\n{'Node':<16} {'GPU':<6} {'Util mean':>10} {'Util max':>10} {'Mem mean (GB)':>14} {'Mem max (GB)':>13}"
    print(hdr)
    print("-" * len(hdr.rstrip()))
    for hostname in sorted(hosts):
        df = hosts[hostname]
        short = hostname.split(".")[0]
        util_cols = sorted(c for c in df.columns if c.endswith("_utilization"))
        mem_cols  = sorted(c for c in df.columns if c.endswith("_memory_used_gb"))
        first = True
        for u_col, m_col in zip(util_cols, mem_cols):
            gpu_id = u_col.split("_")[1]
            node_label = short if first else ""
            print(
                f"{node_label:<16} {'GPU ' + gpu_id:<6} "
                f"{df[u_col].mean():>10.1f} "
                f"{df[u_col].max():>10.1f} "
                f"{df[m_col].mean():>14.2f} "
                f"{df[m_col].max():>13.2f}"
            )
            first = False

    print(f"\nDuration: {duration:.0f}s  ({duration / 60:.1f} min)  |  Samples: {total_samples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot NVML GPU telemetry")
    parser.add_argument("--telemetry-dir", default="nvml-telemetry",
                        help="Directory containing nvml_checkpoint_*.json files")
    parser.add_argument("--output", default="gpu_utilization.png",
                        help="Output plot file")
    args = parser.parse_args()

    hosts = load_telemetry_by_host(args.telemetry_dir)
    print_summary(hosts)
    plot(hosts, args.output)

# python src/plot/plot_nvml.py --telemetry-dir nvml-telemetry --output gpu_util.png
