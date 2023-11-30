#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=PAS1576
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu
#SBATCH --job-name=training
#SBATCH --time=96:00:00
#SBATCH --mem=800GB

source $HOME/projects/open_clip/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3

.venv/bin/torchrun --nproc_per_node 4 -m src.training.main \
  --save-frequency 1 \
  --train-data './data/train/shard-{000000..000159}.tar' \
  --val-data './data/val/shard-{000000..000031}.tar' \
  --dataset-type 'webdataset' \
  --pretrained 'openai' \
  --text_type 'random' \
  --dataset-resampled \
  --warmup 1000 \
  --batch-size 4096 \
  --accum-freq 1 \
  --epochs 100 \
  --workers 8 \
  --model ViT-B-16 \
  --log-every-n-steps 1 \
  --lr 1e-4 \
  --seed 42 \
  --local-loss \
  --gather-with-grad \
  --grad-checkpointing \
  --logs '../storage/log/' \
