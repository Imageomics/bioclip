#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=make-dataset-wds_test
#SBATCH --output=make-dataset-wds_test-%j.out
#SBATCH --error=make-dataset-wds_test-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=serial

echo $SLURM_JOB_NAME

module load miniconda3/24.1.2-py310 # Use your latest miniconda version

conda activate bioclip-train

python scripts/evobio10m/make_wds_reproduce.py --tag CVPR-2024 --split val --workers $SLURM_CPUS_PER_TASK
python scripts/evobio10m/make_wds_reproduce.py --tag CVPR-2024 --split train_small --workers $SLURM_CPUS_PER_TASK
python scripts/evobio10m/make_wds_reproduce.py --tag CVPR-2024 --split train --workers $SLURM_CPUS_PER_TASK
