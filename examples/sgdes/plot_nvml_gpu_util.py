#!/usr/bin/env python3
"""
Plot GPU utilization from nvml-telemetry checkpoints.

Usage:
    python plot_gpu_utilization.py [--telemetry-dir nvml-telemetry] [--output gpu_util.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_telemetry(telemetry_dir: str) -> pd.DataFrame:
    records = []
    for path in sorted(Path(telemetry_dir).glob("nvml_checkpoint_*.json")):
        with open(path) as f:
            data = json.load(f)
        for sample in data["metrics"]:
            row = {"timestamp": sample["timestamp"]}
            m = sample["metrics"]
            for key, val in m.items():
                if key.startswith("gpu_") and key.endswith("_utilization") and "memory" not in key:
                    row[key] = val
                elif key.startswith("gpu_") and key.endswith("_memory_used_gb"):
                    row[key] = val
            records.append(row)

    df = pd.DataFrame(records).drop_duplicates("timestamp").sort_values("timestamp")
    df["time_s"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df


def plot(df: pd.DataFrame, output: str) -> None:
    util_cols = sorted(c for c in df.columns if c.endswith("_utilization"))
    mem_cols  = sorted(c for c in df.columns if c.endswith("_memory_used_gb"))
    n_gpus    = len(util_cols)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # ── GPU compute utilization ───────────────────────────────────────────────
    ax = axes[0]
    for col in util_cols:
        gpu_id = col.split("_")[1]
        ax.plot(df["time_s"], df[col], label=f"GPU {gpu_id}", linewidth=0.8)
    ax.set_ylabel("Utilization (%)")
    ax.set_title(f"GPU Compute Utilization ({n_gpus} GPUs)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=4, fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── GPU memory used ───────────────────────────────────────────────────────
    ax = axes[1]
    for col in mem_cols:
        gpu_id = col.split("_")[1]
        ax.plot(df["time_s"], df[col], label=f"GPU {gpu_id}", linewidth=0.8)
    ax.set_ylabel("Memory Used (GB)")
    ax.set_xlabel("Time (s)")
    ax.set_title("GPU Memory Usage")
    ax.legend(ncol=4, fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved → {output}")


def print_summary(df: pd.DataFrame) -> None:
    util_cols = sorted(c for c in df.columns if c.endswith("_utilization"))
    mem_cols  = sorted(c for c in df.columns if c.endswith("_memory_used_gb"))

    print(f"\n{'GPU':<6} {'Util mean':>10} {'Util max':>10} {'Mem mean (GB)':>14} {'Mem max (GB)':>13}")
    print("-" * 58)
    for u_col, m_col in zip(util_cols, mem_cols):
        gpu_id = u_col.split("_")[1]
        print(
            f"GPU {gpu_id:<2} "
            f"{df[u_col].mean():>10.1f} "
            f"{df[u_col].max():>10.1f} "
            f"{df[m_col].mean():>14.2f} "
            f"{df[m_col].max():>13.2f}"
        )
    duration = df["time_s"].iloc[-1]
    print(f"\nDuration: {duration:.0f}s  ({duration/60:.1f} min)  |  Samples: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry-dir", default="nvml-telemetry")
    parser.add_argument("--output", default="gpu_utilization.png")
    args = parser.parse_args()

    df = load_telemetry(args.telemetry_dir)
    print_summary(df)
    plot(df, args.output)

#python plot_gpu_utilization.py --telemetry-dir nvml-telemetry --output gpu_util.png 