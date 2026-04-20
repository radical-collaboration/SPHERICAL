#!/bin/sh -l

#SBATCH -A dmr170002p
#SBATCH --partition=GPU  #-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#xSBATCH --gpus=v100-32:16
#SBATCH --gpus=8
#SBATCH --time=01:25:00
#SBATCH --job-name=spher
#SBATCH --mail-user=mariya.goliyad@rutgers.edu
#SBATCH --mail-type=ALL

export BASE_DIR="${PROJECT}"
export WORK_DIR="${BASE_DIR}/DeepDriveSim/workflows/ddmd_workflow"
export CONDA_ENV="${BASE_DIR}/conda_env"
export INPUT_DIR="${WORK_DIR}/data"
export INF_DIR
export DDSIM_DIR
export
export

#WARNING: this directory has to be empty before running new experiment!
export EXPRMNT_DIR=$WORK_DIR/ddmd_test_experiments
# Remove the following line if you want to keep data from previous experiments.
rm -rf $EXPRMNT_DIR

cp  $INPUT_DIR/lassen-keras-dbscan.yaml $INPUT_DIR/new_lassen-keras-dbscan.yaml
sed -i "s|\${EXPRMNT_DIR}|$EXPRMNT_DIR|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml 
sed -i "s|\${CONDA_ENV}|$CONDA_ENV|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml
sed -i "s|\${WORK_DIR}|$WORK_DIR|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml

unset SLURM_EXPORT_ENV
module load anaconda3
#module load anaconda
module load cuda/12.6.1
module load cudnn/8.0.4
module load openmpi/5.0.8-gcc13.3.1
source activate base
conda activate $CONDA_ENV/campaign_manager

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_cuda_data_dir=${CUDA_HOME}"

cd $BASE_DIR/htp/SPHERICAL/examples/run_campaign
rm -rf data/telemetry-results
#python run_campaing.py
dragon -s -l debug run_campaing.py
