#!/bin/bash

set -ex
time_start=$(date +%s)

# Source setup script
source "$SLURM_SUBMIT_DIR/scripts/setup_download_tol-10m_components.bash"

# source "scripts/setup_download_tol-10m_components.bash" # Use this for troubleshooting or running this component independently

# Download metadata
hf_metadata_url_prefix="https://huggingface.co/datasets/imageomics/TreeOfLife-10M/resolve/main/metadata/"
hf_naming_url_prefix="https://huggingface.co/datasets/imageomics/TreeOfLife-10M/resolve/main/metadata/naming/"

# Define URLs and output paths
urls_and_paths=(
    "${hf_metadata_url_prefix}catalog.csv?download=true ${metadata_path}/catalog.csv"
    "${hf_metadata_url_prefix}licenses.csv?download=true ${metadata_path}/licenses.csv"
    "${hf_metadata_url_prefix}mapping.sqlite?download=true ${metadata_path}/mapping.sqlite"
    "${hf_metadata_url_prefix}taxon.tab?download=true ${metadata_path}/taxon.tab"
    "${hf_metadata_url_prefix}species_level_taxonomy_chains.csv?download=true ${metadata_path}/species_level_taxonomy_chains.csv"
    "${hf_naming_url_prefix}bioscan_name_lookup.json?download=true ${names_path}/bioscan_name_lookup.json"
    "${hf_naming_url_prefix}inat21_name_lookup.json?download=true ${names_path}/inat21_name_lookup.json"
    "${hf_naming_url_prefix}eol_name_lookup.json?download=true ${names_path}/eol_name_lookup.json"
)

# Download files
for item in "${urls_and_paths[@]}"; do
    read -r url output_path <<< "$item"
    output_dir=$(dirname "$output_path")
    output_file=$(basename "$output_path")
    download "$url" "$output_dir" "$output_file"
done

# End timing
time_end=$(date +%s)
time_elapsed=$((time_end - time_start))

# Notify completion
echo "Total execution time: $time_elapsed seconds" > "${logs_path}/download_metadata_completed.txt"
