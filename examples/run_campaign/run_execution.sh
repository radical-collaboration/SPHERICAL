#!/bin/sh -l

export BASE_DIR="${PROJECT}"
export DDSim_DIR="${BASE_DIR}/DeepDriveSim"
export WORK_DIR="${DDSim_DIR}/pipelines/ddmd_pipeline"
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

unset SLURM_EXPORT_ENV
module load anaconda3
module load anaconda
source activate base
#conda activate   $CONDA_ENV/deepdrivesim

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

conda activate ${BASE_DIR}/conda_env/test_inf
cd ${BASE_DIR}/htp/SPHERICAL/examples/run_campaign

python run_esm2_infern.py

# dragon-network-config --output-to-yaml 

# dragon -w ssh --network-config slurm.yaml run_campaign.py
