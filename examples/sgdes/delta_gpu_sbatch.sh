#!/bin/sh -l

#SBATCH -A ***-delta-gpu 
#xSBATCH -A bebo-delta-gpu 
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=4
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

export SGDES_DIR=/scratch/bblj/mgoliyad1/SGDES
export SPHERICAL_DIR=/scratch/bblj/mgoliyad1/SPHERICAL

cd $SPHERICAL_DIR/examples/sgdes
rm -rf *telemetry
rm -rf mayv_output
rm -rf tmp*

source /u/mgoliyad1/ve/sgdes/bin/activate
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

#ps -ef | grep mgoliyad1 | grep -E "dragon|run_workflow|trill" | awk '{print $2}' | xargs -r kill -9
