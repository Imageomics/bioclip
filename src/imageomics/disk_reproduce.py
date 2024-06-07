"""
All filepaths.
"""

# REMOVE
# eol_vernacularnames_csv = "data/eol/vernacularnames.csv"
# eol_scraped_page_ids_csv = "data/eol/scraped_page_ids.csv"
# eol_taxon_tab = "data/eol/dh21/taxon.tab"

# REMOVE
# itis_hierarchy_csv = (
#     "/fs/ess/PAS2136/open_clip/data/itis/data/interim/species_level_taxonomy_chains.csv"
# )

# REMOVE
# inaturalist_vernacularnames_csv = "data/inat/dwca/VernacularNames-english.csv"
# inaturalist_taxa_csv = "data/inat/dwca/taxa.csv"

# REMOVE
# bioscan_metadata_jsonld = (
#     "/fs/scratch/PAS2136/bioscan/BIOSCAN_Insect_Dataset_metadata.jsonld"
# )

# REMOVE
# resolved_jsonl = "/fs/ess/PAS2136/open_clip/data/resolved.jsonl"

# Actual datasets
DATASET_DIR = "data/TreeOfLife-10M/dataset/"

eol_root_dir = f"{DATASET_DIR}eol"
inat21_root_dir = f"{DATASET_DIR}inat21"
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

# REMOVE
# common_name_lookup_json = "data/names/scientific_to_common.json"
