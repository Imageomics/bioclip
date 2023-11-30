#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --account=PAS2136
#SBATCH --partition=serial

echo $SLURM_JOB_NAME

source $HOME/projects/open_clip/pitzer-venv/bin/activate

python scripts/evobio10m/make_wds.py --tag v3.3-cross-entropy --split val --workers $SLURM_CPUS_PER_TASK
python scripts/evobio10m/make_wds.py --tag v3.3-cross-entropy --split train_small --workers $SLURM_CPUS_PER_TASK
# python scripts/evobio10m/make_wds.py --tag v3.3-cross-entropy --split train --workers $SLURM_CPUS_PER_TASK
