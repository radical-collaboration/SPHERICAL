#!/bin/sh -l
#
# SGDES Campaign — SLURM GPU batch script (Dragon backend)
#
# Account: set SBATCH_ACCOUNT=<project>-delta-gpu before calling sbatch

#SBATCH --partition=gpuA40x4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#xSBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --job-name=sgdes
#SBATCH --mail-user=mariya.goliyad@rutgers.edu
#SBATCH --mail-type=ALL

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TF_FORCE_GPU_ALLOW_GROWTH=true
export JAX_PLATFORMS=cpu

# ── Environment ───────────────────────────────────────────────────────────────
if [ -z "${SBATCH_ACCOUNT:-}${SLURM_JOB_ACCOUNT:-}" ]; then
    echo "WARNING: SBATCH_ACCOUNT is not set — job may be charged to default account."
    echo "         Set it with: export SBATCH_ACCOUNT=<project>-delta-gpu"
fi
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"

if [ -z "${SCRATCH:-}" ]; then
    echo "ERROR: SCRATCH is not set."
    echo "       export SCRATCH=/scratch/<allocation> && sbatch delta_gpu_batch.sh"
    exit 1
fi

export SGDES_DIR="${SGDES_DIR:-${SCRATCH}/${USER}/sgdes}"
export SPHERICAL_DIR="${SPHERICAL_DIR:-${SCRATCH}/${USER}/SPHERICAL}"

cd $SPHERICAL_DIR/workflows/sgdes
# ── Clean previous run artifacts ──────────────────────────────────────────────
rm -rf mayv_output tmp*

# ── Node-local /tmp for foldseek DB files ─────────────────────────────────────
# Pre-create the per-job base dir on every compute node's local /tmp.
# foldseek_createdb (executable_task) and foldseek_search (function_task) are
# pinned to the same node via policy, so both see the same /tmp.
export DES_TMPDIR=/tmp/sgdes_${SLURM_JOBID}
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 mkdir -p $DES_TMPDIR

source /u/${USER}/ve/sgdes/bin/activate
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64

# Compute total GPUs and choose single- vs multi-node Dragon launch.
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}
export TOTAL_GPUS=$(( SLURM_NNODES * GPUS_PER_NODE ))
echo "Nodes: ${SLURM_NNODES}  GPUs/node: ${GPUS_PER_NODE}  Total GPUs: ${TOTAL_GPUS}"

if [ "${SLURM_NNODES}" -gt 1 ]; then
    dragon -m run_workflow.py
else
    dragon -s run_workflow.py
fi

#ps -ef | grep ${USER} | grep -E "dragon|run_workflow|trill" | awk '{print $2}' | xargs -r kill -9
