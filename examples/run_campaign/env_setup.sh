#!/bin/bash
export BASE_DIR="${PROJECT}"
export SPHERICAL_DIR="${BASE_DIR}/htp/SPHERICAL"
export DDSIM_DIR="${BASE_DIR}/DeepDriveSim"
export WORK_DIR="${SPHERICAL_DIR}/examples/run_campaign"
export CONDA_ENV="${BASE_DIR}/conda_env"

#mkdir $CONDA_ENV

module load anaconda3

conda create -y -p $CONDA_ENV/campaign_manager python=3.10
conda activate $CONDA_ENV/campaign_manager
pip install --upgrade pip setuptools wheel
cd $SPHERICAL_DIR
pip install -e ".[dragon,dev,esm2]"
cd  $BASE_DIR
if [ ! -d "$DDSIM_DIR" ]; then
    git clone --branch origin/campaign_manager --single-branch https://github.com/radical-collaboration/DeepDriveSim.git
fi
cd $DDSIM_DIR
pip install -e .
# cd "${DDSIM_DIR}/workflows/dummy_workflow/"
# pip install -r "requirements.txt"
# cd "${DDSIM_DIR}/workflows/ddmd_workflow/"
# pip install -r "requirements.txt"
# cd "${DDSIM_DIR}/workflows/miniapps_workflow/"
# pip install -r "requirements.txt"
cd $WORK_DIR
pip install -r "requirements.txt"
# conda init
conda deactivate