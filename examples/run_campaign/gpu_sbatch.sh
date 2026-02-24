#!/bin/sh -l

#SBATCH -A dmr170002p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#xSBATCH --gpus=v100-32:16
#SBATCH --gpus=4
#SBATCH --time=00:25:00
#SBATCH --job-name=spher
#SBATCH --mail-user=mariya.goliyad@rutgers.edu
#SBATCH --mail-type=ALL

export BASE_DIR="/ocean/projects/dmr170002p/goliyad/htp/SPHERICAL"
export WORK_DIR="${BASE_DIR}//examples/esm2_inference"
export CONDA_ENV="${WORK_DIR}/conda_env"

module load cuda
module load gcc
module load anaconda3
conda activate $(CONDA_ENV)/campaing_manager

cd /ocean/projects/dmr170002p/goliyad/htp/SPHERICAL/examples/campaing_manager

python run_campaign.py

# dragon-network-config --output-to-yaml 

# dragon -w ssh --network-config slurm.yaml run_campaign.py
