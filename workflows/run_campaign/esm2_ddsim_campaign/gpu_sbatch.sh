#!/bin/sh -l

#SBATCH -A *** 
#SBATCH --partition=GPU   #-shared
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#xSBATCH --gpus=v100-32:8
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --export    NONE
#SBATCH --time=02:30:00
#SBATCH --job-name sphr
#SBATCH --mail-user=mg2347@soe.rutgers.edu
#SBATCH --mail-type=ALL      # When to send emails (BEGIN, END, FAIL, ALL)


export BASE_DIR="${PROJECT}"
export WORK_DIR="${BASE_DIR}/DeepDriveSim/workflows/ddmd_workflow"
export CONDA_ENV="${BASE_DIR}/conda_env"
export INPUT_DIR="${WORK_DIR}/data"
export DUMMY_DIR="${BASE_DIR}/DeepDriveSim/workflows/dummy_workflow"
export INF_DIR="${BASE_DIR}/htp/SPHERICAL/workflows/esm2_inference"
export MINAPPS_DIR="${BASE_DIR}/DeepDriveSim/workflows/miniapps_workflow"
export MD_DIR="${WORK_DIR}"

#WARNING: this directory has to be empty before running new experiment!
export EXPRMNT_DIR=$WORK_DIR/ddmd_test_experiments
# Remove the following line if you want to keep data from previous experiments.
rm -rf $EXPRMNT_DIR

unset SLURM_EXPORT_ENV
module load anaconda3
#module load anaconda
source activate base
conda activate   $CONDA_ENV/campaign_manager

export CUDA_HOME=/opt/packages/cuda/v12.6.1
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TF_FORCE_GPU_ALLOW_GROWTH=true

cp  $INPUT_DIR/lassen-keras-dbscan.yaml $INPUT_DIR/new_lassen-keras-dbscan.yaml
sed -i "s|\${EXPRMNT_DIR}|$EXPRMNT_DIR|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml
sed -i "s|\${CONDA_ENV}|$CONDA_ENV|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml
sed -i "s|\${WORK_DIR}|$WORK_DIR|g" $INPUT_DIR/new_lassen-keras-dbscan.yaml

cp  $WORK_DIR/template_config.yaml $WORK_DIR/config.yaml
sed -i "s|\${PROJECT}|$PROJECT|g" $WORK_DIR/config.yaml

# NOTE: the campaign config (esm2_ddsim_campaign/config.yaml) is committed and
# self-templating — its ${MD_DIR}/${MINAPPS_DIR}/${INF_DIR}/${DUMMY_DIR} refs are
# expanded at load time by _expand_env(), so no cp/sed step is needed here.
# (The per-workflow configs below ARE generated, because the campaign config
# points at them and the workflows read them directly.)

cp  $MINAPPS_DIR/template_config.yaml $MINAPPS_DIR/config.yaml
sed -i "s|\${PROJECT}|$PROJECT|g" $MINAPPS_DIR/config.yaml
cp  $DUMMY_DIR/template_config.yaml $DUMMY_DIR/config.yaml
sed -i "s|\${PROJECT}|$PROJECT|g" $DUMMY_DIR/config.yaml

# Run from the campaign directory where run_campaing.py + config.yaml live, so
# the workflow_registry modules import and the relative telemetry_dir resolves here.
cd $BASE_DIR/htp/SPHERICAL/workflows/run_campaign/esm2_ddsim_campaign

# Clear telemetry from a previous run (telemetry_dir is relative to this dir).
rm -rf telemetry-results nvml-telemetry

dragon run_campaing.py --config config.yaml
# Local (no Dragon):  python run_campaing.py --config config.yaml --engine concurrent
