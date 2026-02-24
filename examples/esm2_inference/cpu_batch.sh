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

module load cuda
module load gcc
module load anaconda3
conda activate /ocean/projects/dmr170002p/goliyad/conda_env/test_inf

cd /ocean/projects/dmr170002p/goliyad/htp/SPHERICAL/examples/esm2_inference

#dragon-network-config --output-to-yaml 

#dragon -w ssh --network-config slurm.yaml run_esm2_infern.py --config_file config.yaml

python run_esm2_infern.py 