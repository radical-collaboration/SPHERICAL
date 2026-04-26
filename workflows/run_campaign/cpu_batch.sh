#!/bin/sh -l

#SBATCH -A ***
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:25:00
#SBATCH --job-name=spher
#SBATCH --mail-user=mariya.goliyad@rutgers.edu
#SBATCH --mail-type=ALL


export HF_TOKEN="hf_***"

export BASE_DIR="${PROJECT}/DeepDriveSim"
export WORK_DIR="${BASE_DIR}/pipelines/ddmd_pipeline"
export CONDA_ENV="${WORK_DIR}/conda_env"
export INPUT_DIR="${WORK_DIR}/data"

#WARNING: this directory has to be empty before running new experiment!
export EXPRMNT_DIR=$WORK_DIR/ddmd_test_experiments
# Remove the following line if you want to keep data from previous experiments.
rm -rf $EXPRMNT_DIR

cp  $INPUT_DIR/lassen-keras-dbscan.yaml $INPUT_DIR/new_lassen-keras-dbscan.yaml
sed -i "s|\${EXPRMNT_DIR}|$EXPRMNT_DIR|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml 
sed -i "s|\${CONDA_ENV}|$CONDA_ENV|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml
sed -i "s|\${WORK_DIR}|$WORK_DIR|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml

# module load cuda
# module load gcc
# module load anaconda3
#conda activate $(CONDA_ENV)/campaing_manager

unset SLURM_EXPORT_ENV
module load anaconda3
module load anaconda
source activate base
#conda activate   $CONDA_ENV/deepdrivesim

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

conda activate $PROJECT/conda_env/test_inf
cd $PROJECT/htp/SPHERICAL/workflows/run_campaign

python run_esm2_infern.py