"""
All filepaths.
"""

# Actual datasets
DATASET_DIR = "data/TreeOfLife-10M/dataset/"

eol_root_dir = f"{DATASET_DIR}eol"
inat21_root_dir = f"{DATASET_DIR}inat21/train"
bioscan_root_dir = f"{DATASET_DIR}bioscan"

# rare species
seen_in_training_json = "data/rarespecies/seen_in_training.json"
unseen_in_training_json = "data/rarespecies/unseen_in_training.json"

# Files we make
METADATA_DIR = "data/TreeOfLife-10M/metadata/"

eol_name_lookup_json = f"{METADATA_DIR}naming/eol_name_lookup.json"
inat21_name_lookup_json = f"{METADATA_DIR}naming/inat21_name_lookup.json"
bioscan_name_lookup_json = f"{METADATA_DIR}naming/bioscan_name_lookup.json"

db = f"{METADATA_DIR}mapping.sqlite"
