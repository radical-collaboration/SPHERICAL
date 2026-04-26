#!/bin/sh -l

#SBATCH -A *** 
#SBATCH --partition=GPU  #-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH  --gpus=8
#xSBATCH --exclusive
#SBATCH --time=02:30:00
#SBATCH --job-name=sgdes
#SBATCH --mail-user=mariya.goliyad@rutgers.edu
#SBATCH --mail-type=ALL


export CONDA_ENV="${PROJECT}/conda_env"

# unset SLURM_EXPORT_ENV
module load anaconda3 || true
module load cuda/12.6.1 || true
module load cudnn/8.0.4 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV/sgdes

export CUDA_HOME=/opt/packages/cuda/v12.6.1
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export SGDES_DIR=$PROJECT/sgdes/SGDES
export SPHERICAL_DIR=$PROJECT/htp/SPHERICAL

cd $SPHERICAL_DIR/workflows/sgdes
rm -rf nvml-telemetry
rm -rf mayv_output
rm -rf tmp*

$CONDA_ENV/sgdes/bin/dragon -s run_workflow.py
