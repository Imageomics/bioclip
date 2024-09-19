#!/bin/bash
#SBATCH --job-name=make_catalog_%j 
#SBATCH --output=logs/make_catalog_%j.out
#SBATCH --error=logs/make_catalog_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1

# Usage:
# sbatch --account <HPC-account> --cpus-per-task <N> slurm/make-catalog_reproduce.slurm \
# --dir <path/to/splits> \
# --db <path/to/db> \
# --tag <tag> \
# --batch-size <batch-size>
# 
# e.g.
# 
# sbatch --account ABC123 --cpus-per-task 32 slurm/make-catalog_reproduce.slurm \
# --dir data/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224 \
# --db data/TreeOfLife-10M/metadata/mapping.sqlite \
# --tag CVPR-2024 \
# --batch-size 256

module load miniconda3/24.1.2-py310 # Adjust as needed

conda activate bioclip-train

python scripts/evobio10m/make_catalog_reproduce.py --workers "$SLURM_CPUS_PER_TASK" "$@"
