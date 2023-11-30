#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export DATA_ROOT="/local/scratch/imageomics/projects/open_clip/data/rare_species"
export PRETRAINED="/local/scratch/imageomics/projects/open_clip/model/10m_random/epoch_100.pt"
# "laion400m_e32" "openai"
#"/local/scratch/imageomics/projects/open_clip/model/10m_random/epoch_100.pt"
#"/local/scratch/imageomics/projects/open_clip/model/inat_random/epoch_65.pt"
export TEXT_TYPE="taxon_com"
#"com", "sci", "sci_com", "taxon_com"
export LABEL_FILE="metadata.csv"
export LOG_FILEPATH="../storage/logs"

python -m src.evaluation.zero_shot_iid \
        --model "ViT-B-16" \
        --batch-size 256 \
        --data_root $DATA_ROOT \
        --pretrained $PRETRAINED \
        --label_filename $LABEL_FILE \
        --log $LOG_FILEPATH \
        --text_type $TEXT_TYPE \


# conda activate bioclip-test
# ./eval_zero_shot.sh &> ../storage/zeroshot_10m_random.txt
