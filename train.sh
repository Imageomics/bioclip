#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node 4 -m src.training.main \
  --train-data './data/train/shard-{000000..000159}.tar' \
  --val-data './data/val/shard-{000000..000031}.tar' \
  --dataset-type 'webdataset' \
  --pretrained 'openai' \
  --text_type 'random' \
  --warmup 1000 \
  --batch-size 4096 \
  --accum-freq 1 \
  --epochs 100 \
  --workers 8 \
  --model ViT-B-16 \
  --lr 1e-4 \
  --log-every-n-steps 1 \
  --dataset-resampled \
  --local-loss \
  --gather-with-grad \
  --grad-checkpointing \
  --logs '../storage/log/' \
