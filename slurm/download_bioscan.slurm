#!/bin/bash

set -ex
time_start=$(date +%s)

# Source setup script
source "$SLURM_SUBMIT_DIR/scripts/setup_download_tol-10m_components.bash"

# source "scripts/setup_download_tol-10m_components.bash" # Use this for troubleshooting or running this component independently

# Download BIOSCAN data
bioscan_url="https://zenodo.org/record/8030065/files/cropped_256.zip?download=1"
download "$bioscan_url" "$bioscan_root" "cropped_256.zip"

# Extract BIOSCAN archive
zip_archive="${bioscan_root}/cropped_256.zip"
extract_bioscan "$zip_archive" "$bioscan_root"

# End timing
time_end=$(date +%s)
time_elapsed=$((time_end - time_start))

# Notify completion
echo "Total execution time: $time_elapsed seconds" > "${logs_path}/download_bioscan_completed.txt"
