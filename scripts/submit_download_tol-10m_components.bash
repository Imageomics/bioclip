#!/bin/bash

set -ex

# Usage: sbatch --account <your-account> submit_download_tol-10m_components.bash

# Source the setup script
source "setup_download_tol-10m_components.bash"

# Ensure necessary directories exist
mkdir -p "$logs_path"

# Submit jobs
metadata_job=$(sbatch \
    --account=$SLURM_JOB_ACCOUNT \
    --time=$METADATA_SLURM_TIME \
    --nodes=$SLURM_NODES \
    --ntasks-per-node=$METADATA_SLURM_NTASKS_PER_NODE \
    --output=$METADATA_SLURM_OUTPUT \
    --error=$METADATA_SLURM_ERROR \
    "$SLURM_SUBMIT_DIR/download_metadata.slurm" | awk '{print $4}')

eol_job=$(sbatch \
    --account=$SLURM_JOB_ACCOUNT \
    --time=$EOL_SLURM_TIME \
    --nodes=$SLURM_NODES \
    --ntasks-per-node=$EOL_SLURM_NTASKS_PER_NODE \
    --output=$EOL_SLURM_OUTPUT \
    --error=$EOL_SLURM_ERROR \
    "$SLURM_SUBMIT_DIR/download_eol.slurm" | awk '{print $4}')

inat21_job=$(sbatch \
    --account=$SLURM_JOB_ACCOUNT \
    --time=$INAT21_SLURM_TIME \
    --nodes=$SLURM_NODES \
    --ntasks-per-node=$INAT21_SLURM_NTASKS_PER_NODE \
    --output=$INAT21_SLURM_OUTPUT \
    --error=$INAT21_SLURM_ERROR \
    "$SLURM_SUBMIT_DIR/download_inat21.slurm" | awk '{print $4}')

bioscan_job=$(sbatch \
    --account=$SLURM_JOB_ACCOUNT \
    --time=$BIOSCAN_SLURM_TIME \
    --nodes=$SLURM_NODES \
    --ntasks-per-node=$BIOSCAN_SLURM_NTASKS_PER_NODE \
    --output=$BIOSCAN_SLURM_OUTPUT \
    --error=$BIOSCAN_SLURM_ERROR \
    "$SLURM_SUBMIT_DIR/download_bioscan.slurm" | awk '{print $4}')

echo "Submitted jobs: metadata ($metadata_job), EOL ($eol_job), iNat21 ($inat21_job), BIOSCAN ($bioscan_job)"

echo "Script dir: $SCRIPT_DIR"
echo "Repo root: $REPO_ROOT"
