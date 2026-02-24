#!/bin/sh -l

#SBATCH -A dmr170002p
#SBATCH --partition=RM
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=01:25:00
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

#dragon-network-config --output-to-yaml 

#dragon -w ssh --network-config slurm.yaml run_esm2_infern.py --config_file config.yaml

python run_campaign.py