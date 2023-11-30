"""
All filepaths.
"""

eol_vernacularnames_csv = "data/eol/vernacularnames.csv"
eol_scraped_page_ids_csv = "data/eol/scraped_page_ids.csv"
eol_taxon_tab = "data/eol/dh21/taxon.tab"

itis_hierarchy_csv = (
    "/fs/ess/PAS2136/open_clip/data/itis/data/interim/species_level_taxonomy_chains.csv"
)

inaturalist_vernacularnames_csv = "data/inat/dwca/VernacularNames-english.csv"
inaturalist_taxa_csv = "data/inat/dwca/taxa.csv"

bioscan_metadata_jsonld = (
    "/fs/scratch/PAS2136/bioscan/BIOSCAN_Insect_Dataset_metadata.jsonld"
)

resolved_jsonl = "/fs/ess/PAS2136/open_clip/data/resolved.jsonl"

# Actual datasets
eol_root_dir = "/fs/ess/PAS2136/eol/data/interim/media_cargo_archive"
inat21_root_dir = "/fs/ess/PAS2136/foundation_model/inat21/raw/train"
bioscan_root_dir = "/fs/scratch/PAS2136/bioscan/cropped_256"

# rare species
seen_in_training_json = "data/rarespecies/seen_in_training.json"
unseen_in_training_json = "data/rarespecies/unseen_in_training.json"

# Files we make

eol_name_lookup_json = "data/names/eol_name_lookup.json"
inat21_name_lookup_json = "data/names/inat21_name_lookup.json"
bioscan_name_lookup_json = "data/names/bioscan_name_lookup.json"

common_name_lookup_json = "data/names/scientific_to_common.json"
