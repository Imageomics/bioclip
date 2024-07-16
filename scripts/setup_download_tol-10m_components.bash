#!/bin/bash

REPO_ROOT="$PWD"
SLURM_SUBMIT_DIR="$REPO_ROOT/slurm"

export REPO_ROOT SLURM_SUBMIT_DIR

# IF YOU WOULD LIKE TO SET A CUSTOM DATASET PATH OR SLURM DIRECTIVES, CHANGE THE VARIABLES BELOW


####################################### customize-below >>>


# optional:
# Set a custom path for the dataset. 
# It defaults to the location below within the git repository.
# If you leave the default location, ensure you have placed the repository in a location with sufficient space.
dataset_path="${REPO_ROOT}/data/TreeOfLife-10M"

SLURM_NODES=1

# SLURM directives for each job
# Wall times are set conservatively to ensure jobs complete on most systems

# Quick download of metadata files
METADATA_SLURM_TIME="0:20:00"
METADATA_SLURM_NTASKS_PER_NODE=1
METADATA_SLURM_OUTPUT="${REPO_ROOT}/logs/download_metadata_%j.out"
METADATA_SLURM_ERROR="${REPO_ROOT}/logs/download_metadata_%j.err"

# Network-intenstive download of 63x ~30GB .tar.gz files with EOL images (~2.5h on OSC, 200-300MiB/s)
# Wall time may need to increase if `wget` is used instead of `aria2c`
EOL_SLURM_TIME="18:00:00"
EOL_SLURM_NTASKS_PER_NODE=12
EOL_SLURM_OUTPUT="${REPO_ROOT}/logs/download_eol_%j.out"
EOL_SLURM_ERROR="${REPO_ROOT}/logs/download_eol_%j.err"

# Network intensive download (~11min on OSC, 340MiB/s), CPU-bound extraction (~88min total on OSC)
INAT21_SLURM_TIME="4:00:00"
INAT21_SLURM_NTASKS_PER_NODE=1
INAT21_SLURM_OUTPUT="${REPO_ROOT}/logs/download_inat21_%j.out"
INAT21_SLURM_ERROR="${REPO_ROOT}/logs/download_inat21_%j.err"

# Modest download, CPU-bound extraction (made fast by parallelization) (~8min on OSC)
BIOSCAN_SLURM_TIME="2:00:00"
BIOSCAN_SLURM_NTASKS_PER_NODE=20
BIOSCAN_SLURM_OUTPUT="${REPO_ROOT}/logs/download_bioscan_%j.out"
BIOSCAN_SLURM_ERROR="${REPO_ROOT}/logs/download_bioscan_%j.err"


####################################### customize-above <<<


export SLURM_ACCOUNT SLURM_NODES
export METADATA_SLURM_TIME METADATA_SLURM_NTASKS_PER_NODE METADATA_SLURM_OUTPUT METADATA_SLURM_ERROR
export EOL_SLURM_TIME EOL_SLURM_NTASKS_PER_NODE EOL_SLURM_OUTPUT EOL_SLURM_ERROR
export INAT21_SLURM_TIME INAT21_SLURM_NTASKS_PER_NODE INAT21_SLURM_OUTPUT INAT21_SLURM_ERROR
export BIOSCAN_SLURM_TIME BIOSCAN_SLURM_NTASKS_PER_NODE BIOSCAN_SLURM_OUTPUT BIOSCAN_SLURM_ERROR

metadata_path="${dataset_path}/metadata"
names_path="${metadata_path}/naming"
eol_root="${dataset_path}/dataset/eol"
inat21_root="${dataset_path}/dataset/inat21"
bioscan_root="${dataset_path}/dataset/bioscan"
rarespecies_root="${dataset_path}/dataset/rarespecies"
container_path="${REPO_ROOT}/containers"
logs_path="${REPO_ROOT}/logs"

export dataset_path metadata_path names_path eol_root inat21_root bioscan_root rarespecies_root container_path logs_path

# Create directories
mkdir -p "${dataset_path}" "${metadata_path}" "${names_path}" "${eol_root}" "${inat21_root}" "${bioscan_root}" "${rarespecies_root}" "${container_path}" "${logs_path}"

is_slurm_job() {
    [ -n "$SLURM_JOB_ID" ]
}

# Download function for fast download with aria2c if Apptainer is available, otherwise uses wget
download() {
    echo "Downloading $1 to $2/$3"
    url=$1
    output_path=$2
    output_file=$3

    if [ -n "$aria2_sif_path" ] && [ -f "$aria2_sif_path" ]; then
        apptainer exec "$aria2_sif_path" aria2c \
            --dir="$output_path" \
            --out="$output_file" \
            --continue \
            --max-connection-per-server=16 \
            "$url"
        if [ $? -eq 0 ]; then
            return 0
        fi
    fi

    wget -c -O "${output_path}/${output_file}" "$url"
}

export -f download

# Function to extract BIOSCAN-1M
extract_bioscan() {
    local zip_file="$1"
    local output_dir="$2"
    local temp_dir="${output_dir}/temp"
    local final_file_count=1128313

    mkdir -p "$temp_dir"

    # Adapt resource usage based on environment being used
    local available_cores=$(nproc)
    if is_slurm_job; then
        # Use at least 1
        local requested_tasks=${BIOSCAN_SLURM_NTASKS_PER_NODE:-1}
        requested_tasks=$(( requested_tasks < 1 ? 1 : requested_tasks ))
        
        # Use the smaller of number of cores requested vs available
        local num_tasks=$(( requested_tasks < available_cores ? requested_tasks : available_cores ))
    else
        # For command line, use all available cores up to a maximum of 4
        local num_tasks=$(( available_cores < 4 ? available_cores : 4 ))
    fi

    # Extract all parts in parallel
    total_parts=$(unzip -Z1 "$zip_file" | grep 'bioscan/images/cropped_256/part' | cut -d'/' -f4 | sort -u | wc -l)
    current_part=0
    # Prevent race to create temp parent directories during unzip
    mkdir -p "$temp_dir/bioscan/images/cropped_256/"
    unzip -Z1 "$zip_file" | grep 'bioscan/images/cropped_256/part' | cut -d'/' -f4 | sort -u | \
        xargs -P "$num_tasks" -I {} sh -c '
            unzip -q "$0" "bioscan/images/cropped_256/{}/*" -d "$1"
            echo "Extracted part {}"
        ' "$zip_file" "$temp_dir"

    # Move all part directories to the final location
    find "$temp_dir/bioscan/images/cropped_256/" -type d -name 'part*' -print0 | xargs -0 -I {} mv {} "$output_dir/"

    # Remove the temp directory
    rm -rf "$temp_dir"

    # Count the number of extracted files and verify
    local extracted_file_count
    extracted_file_count=$(find "$output_dir" -type f -name '*.jpg' | wc -l)
    
    if [ "$extracted_file_count" -ne "$final_file_count" ]; then
        echo "Error: Expected $final_file_count files, but found $extracted_file_count files." >&2
        return 1
    fi

    rm "$zip_file"

    echo "Extracted BIOSCAN-1M to $output_dir"
}

export -f extract_bioscan

# Function to extract iNat21
extract_inat21() {
    local tar_archive="$1"
    local output_dir="$2"

    tar -xzf "$tar_archive" -C "$output_dir"

    rm "$tar_archive"
    echo "Extracted iNat21 to $output_dir"
}

export -f extract_inat21

# Container setup for aria2c if Apptainer is available
aria2_tag="202209060423"
aria2_sif_path="${container_path}/aria2-pro_${aria2_tag}.sif"

if command -v apptainer &> /dev/null; then
    mkdir -p "$container_path"
    if [ ! -f "$aria2_sif_path" ]; then
        apptainer build "$aria2_sif_path" "docker://p3terx/aria2-pro:${aria2_tag}"
        if [ $? -ne 0 ]; then
            echo "Failed to build the aria2 container. Falling back to wget for downloads."
            aria2_sif_path=""
        fi
    fi
else
    echo "Apptainer is not installed. Falling back to wget for all downloads."
    aria2_sif_path=""
fi

export aria2_sif_path
