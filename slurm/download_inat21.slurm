#!/bin/bash

set -ex
time_start=$(date +%s)

# Source setup script
source "$SLURM_SUBMIT_DIR/scripts/setup_download_tol-10m_components.bash"

# source "scripts/setup_download_tol-10m_components.bash" # Use this for troubleshooting or running this component independently

# Download iNat21 data
inat21_url="https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz"
download "$inat21_url" "$inat21_root" "train.tar.gz"

# Extract iNat21 archive
tar_archive="${inat21_root}/train.tar.gz"
extract_inat21 "$tar_archive" "$inat21_root"

# End timing
time_end=$(date +%s)
time_elapsed=$((time_end - time_start))

# Notify completion
echo "Total execution time: $time_elapsed seconds" > "${logs_path}/download_inat21_completed.txt"
