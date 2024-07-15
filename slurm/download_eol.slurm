#!/bin/bash

set -ex
time_start=$(date +%s)

# Source setup script
source "$SLURM_SUBMIT_DIR/scripts/setup_download_tol-10m_components.bash"

# source "scripts/setup_download_tol-10m_components.bash" # Use this for troubleshooting or running this component independently

# Download EOL data
for i in $(seq -w 01 63); do
    file="image_set_${i}.tar.gz"
    url="https://huggingface.co/datasets/imageomics/TreeOfLife-10M/resolve/main/dataset/EOL/${file}?download=true"
    download "$url" "$eol_root" "$file"
done

# End timing
time_end=$(date +%s)
time_elapsed=$((time_end - time_start))

# Notify completion
echo "Total execution time: $time_elapsed seconds" > "${logs_path}/download_eol_completed.txt"
