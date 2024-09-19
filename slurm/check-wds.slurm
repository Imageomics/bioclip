#!/bin/bash
#SBATCH --job-name=check_wds
#SBATCH --output=logs/check_wds_%j.out
#SBATCH --error=logs/check_wds_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1

# Usage:
# sbatch --account ACCOUNT --cpus-per-task N slurm/check-wds.slurm SHARDS 
# e.g.
# sbatch --account ABC123 --cpus-per-task 32 slurm/check-wds.slurm 'data/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/train/shard-{000000..000165}.tar'

module load miniconda3/24.1.2-py310 # Adjust as needed

conda activate bioclip-train

shardlist=$1

if [ -z "$shardlist" ]; then
  echo "Shard list is required. Usage: sbatch --account ACCOUNT slurm/check_wds.slurm SHARDS"
  exit 1
fi

srun python scripts/evobio10m/check_wds.py --shardlist "$shardlist" --workers $SLURM_CPUS_PER_TASK > logs/bad-shards.txt
