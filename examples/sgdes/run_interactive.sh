#!/bin/bash
# =============================================================================
# Run SGDES workflow on an already-allocated interactive node.
#
# Usage:
#   bash run_interactive.sh [--gpus N] [--config FILE]
#
# Defaults:
#   --gpus   1               (GPUs visible on this node)
#   --config config.yaml
#
# Typical interactive allocation before running this script:
#   srun --account=bblj-delta-gpu --partition=gpuA40x4 \
#        --nodes=1 --gpus-per-node=4 --cpus-per-task=64 \
#        --time=01:00:00 --pty bash
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
GPUS=1
CONFIG="config.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)   GPUS="$2";   shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export SGDES_DIR=/scratch/bblj/mgoliyad1/SGDES
export SPHERICAL_DIR=/scratch/bblj/mgoliyad1/SPHERICAL

cd "${SPHERICAL_DIR}/examples/sgdes"

# ── Clean previous run artifacts ──────────────────────────────────────────────
rm -rf *telemetry mayv_output tmp*

# ── Activate venv ─────────────────────────────────────────────────────────────
source /u/mgoliyad1/ve/sgdes/bin/activate
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64

# ── Launch ────────────────────────────────────────────────────────────────────
export TOTAL_GPUS=${GPUS}
echo "GPUs: ${TOTAL_GPUS}  Config: ${CONFIG}"

dragon -s run_workflow.py --config "${CONFIG}"
