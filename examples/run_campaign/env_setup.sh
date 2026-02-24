#!/bin/bash
export BASE_DIR="/ocean/projects/dmr170002p/goliyad/htp/SPHERICAL"
export WORK_DIR="${BASE_DIR}//examples/esm2_inference"
export CONDA_ENV="${WORK_DIR}/conda_env"
export DDSIM_DIR="/ocean/projects/dmr170002p/goliyad/DeepDriveSim"

#mkdir $CONDA_ENV

module load anaconda3


##############################################
# 1. DeepDriveSim base env
##############################################
conda create -y -p $CONDA_ENV/dummy_pipeline python=3.9
conda activate $CONDA_ENV/campaing_manager
pip install --upgrade pip setuptools wheel
cd $BASE_DIR
pip install -e .
cd $DDSIM_DIR
pip install -e .
cd $WORK_DIR
pip install -r "requirements.txt"
