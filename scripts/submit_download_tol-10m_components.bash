#!/bin/bash

set -ex

# Run from the root of the repository.
# Usage: sbatch --account <your-account> scripts/submit_download_tol-10m_components.bash

# Source the setup script
source "scripts/setup_download_tol-10m_components.bash"
export SBATCH_ACCOUNT=$SLURM_JOB_ACCOUNT # Applies to all child jobs

# Ensure necessary directories exist
mkdir -p "$logs_path"

# Submit jobs
metadata_job=$(sbatch \
    --time=$METADATA_SLURM_TIME \
    --nodes=$SLURM_NODES \
    --ntasks-per-node=$METADATA_SLURM_NTASKS_PER_NODE \
    --output=$METADATA_SLURM_OUTPUT \
    --error=$METADATA_SLURM_ERROR \
    "$SLURM_SUBMIT_DIR/download_metadata.slurm" | awk '{print $4}')

eol_job=$(sbatch \
    --time=$EOL_SLURM_TIME \
    --nodes=$SLURM_NODES \
    --ntasks-per-node=$EOL_SLURM_NTASKS_PER_NODE \
    --output=$EOL_SLURM_OUTPUT \
    --error=$EOL_SLURM_ERROR \
    "$SLURM_SUBMIT_DIR/download_eol.slurm" | awk '{print $4}')

inat21_job=$(sbatch \
    --time=$INAT21_SLURM_TIME \
    --nodes=$SLURM_NODES \
    --ntasks-per-node=$INAT21_SLURM_NTASKS_PER_NODE \
    --output=$INAT21_SLURM_OUTPUT \
    --error=$INAT21_SLURM_ERROR \
    "$SLURM_SUBMIT_DIR/download_inat21.slurm" | awk '{print $4}')

bioscan_job=$(sbatch \
    --time=$BIOSCAN_SLURM_TIME \
    --nodes=$SLURM_NODES \
    --ntasks-per-node=$BIOSCAN_SLURM_NTASKS_PER_NODE \
    --output=$BIOSCAN_SLURM_OUTPUT \
    --error=$BIOSCAN_SLURM_ERROR \
    "$SLURM_SUBMIT_DIR/download_bioscan.slurm" | awk '{print $4}')

echo "Submitted jobs: metadata ($metadata_job), EOL ($eol_job), iNat21 ($inat21_job), BIOSCAN ($bioscan_job)"

echo "Repo root: $REPO_ROOT"
