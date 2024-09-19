#!/bin/bash
# This script is used to reproduce the TreeOfLife-10M dataset in full

# Set the dataset paths
dataset_path="../data/TreeOfLife-10M"
metadata_path="${dataset_path}/metadata"
names_path="${metadata_path}/naming"
eol_root="${dataset_path}/dataset/eol"
inat21_root="${dataset_path}/dataset/inat21"
bioscan_root="${dataset_path}/dataset/bioscan"

mkdir -p "${dataset_path}" "${metadata_path}" "${names_path}" "${eol_root}" "${inat21_root}" "${bioscan_root}"

# Use aria2c if Apptainer is available for better download performance 
aria2_tag="202209060423"
aria2_sif_path="../containers/aria2-pro_${aria2_tag}.sif"

# Ensure the aria2 container is available
if command -v apptainer &> /dev/null; then
    if [ ! -f "$aria2_sif_path" ]; then
        mkdir -p "../containers"
        apptainer build "$aria2_sif_path" "docker://p3terx/aria2-pro:${aria2_tag}"
        if [ $? -ne 0 ]; then
            echo "Failed to build the aria2 container. Falling back to wget for downloads."
            aria2_sif_path=""  # Ensure the download function knows aria2c is unavailable
        fi
    fi
else
    echo "Apptainer is not installed. Falling back to wget for all downloads."
    aria2_sif_path=""
fi

# Download function that uses aria2c if available
download() {
    echo "Downloading $1 to $2/$3"
    url=$1
    output_path=$2
    output_file=$3

    if [ -n "$aria2_sif_path" ] && [ -f "$aria2_sif_path" ]; then
        # Try downloading with aria2c
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

    # Fallback to wget if aria2c fails or is not available
    wget -c -O "${output_path}/${output_file}" "$url"
}

###TESTOUTFORBIOSCAN
# # Download TOL-10M metadata from Hugging Face
# hf_metadata_url_prefix = "https://huggingface.co/datasets/imageomics/TreeOfLife-10M/resolve/main/metadata/"

# download "${hf_metadata_url_prefix}catalog.csv?download=true" "$metadata_path" "catalog.csv"
# download "${hf_metadata_url_prefix}licenses.csv?download=true" "$metadata_path" "licenses.csv"
# download "${hf_metadata_url_prefix}species_level_taxonomy_chains.csv?download=true" "$metadata_path" "species_level_taxonomy_chains.csv"
# download "${hf_metadata_url_prefix}species_level_taxonomy.csv?download=true" "$metadata_path" "species_level_taxonomy.csv"

# download "${hf_metadata_url_prefix}bioscan_name_lookup.json?download=true" "$names_path" "bioscan_name_lookup.json"
# download "${hf_metadata_url_prefix}inat21_name_lookup.json?download=true" "$names_path" "inat21_name_lookup.json"
# download "${hf_metadata_url_prefix}eol_name_lookup.json?download=true" "$names_path" "eol_name_lookup.json"

# Download the EOL portion from Hugging Face
for i in $(seq -w 22 63); do
    file="image_set_${i}.tar.gz"
    url="https://huggingface.co/datasets/imageomics/TreeOfLife-10M/resolve/main/dataset/EOL/${file}?download=true"

    download "$url" "$eol_root" "$file"
done

# # Download the iNat21 portion
# download "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz" "$inat21_root" "train.tar.gz"
# tar -xzf "${inat21_root}/train.tar.gz" -C "$inat21_root" && rm "${inat21_root}/train.tar.gz" &
###TESTOUTFORBIOSCAN

# Download the BioSCAN-1M portion
bioscan_url="https://zenodo.org/record/8030065/files/cropped_256.zip?download=1"
echo "Downloading BioSCAN-1M"
download "https://zenodo.org/records/8030065/files/cropped_256.zip?download=1" "$bioscan_root" "cropped_256.zip"

####################
##EITHERTHIS
# unzip -q "${bioscan_root}/cropped_256.zip" -d "$bioscan_root" && rm "${bioscan_root}/cropped_256.zip" &
##EITHERTHIS

##ORTHIS
# Extract to a temporary directory
echo "Extracting BioSCAN-1M to $bioscan_root"
temp_extraction_path="${bioscan_root}/temp"
mkdir -p "$temp_extraction_path"
unzip -q "${bioscan_root}/cropped_256.zip" -d "$temp_extraction_path" && \
mv "${temp_extraction_path}/bioscan/images/cropped_256/"* "${bioscan_root}/" && \
rm -r "$temp_extraction_path"

# Rename the directories as needed
cd "${bioscan_root}"
for dir in part*; do
    new_dir_name=$(echo "$dir" | sed 's/part/part/')
    mv "$dir" "$new_dir_name"
done

# Clean up the zip file
rm "${bioscan_root}/cropped_256.zip"
##ORTHIS
####################

# /fs/scratch/PAS2136/thompsonmj/bioclip/data/TreeOfLife-10M/dataset/bioscan/bioscan/images/cropped_256/partN
# https://chatgpt.com/share/053a4e09-a9ad-48e1-b1a6-219421d29aed

