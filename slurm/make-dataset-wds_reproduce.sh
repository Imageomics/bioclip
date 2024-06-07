#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=make-dataset-wds_test
#SBATCH --output=make-dataset-wds_test-%j.out
#SBATCH --error=make-dataset-wds_test-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --account=PAS2136
#SBATCH --partition=serial

echo $SLURM_JOB_NAME

# source $HOME/projects/open_clip/pitzer-venv/bin/activate
module load miniconda3/24.1.2-py310
conda --version
which conda

conda activate /users/PAS2136/thompsonmj/.conda/envs/bioclip-train
pip install -e /fs/scratch/PAS2136/thompsonmj/bioclip
which python
whoami

~/.conda/envs/bioclip-train/bin/python scripts/evobio10m/make_wds_reproduce.py --tag v3.3-cross-entropy --split val --workers $SLURM_CPUS_PER_TASK
~/.conda/envs/bioclip-train/bin/python scripts/evobio10m/make_wds_reproduce.py --tag v3.3-cross-entropy --split train_small --workers $SLURM_CPUS_PER_TASK
~/.conda/envs/bioclip-train/bin/python scripts/evobio10m/make_wds_reproduce.py --tag v3.3-cross-entropy --split train --workers $SLURM_CPUS_PER_TASK
