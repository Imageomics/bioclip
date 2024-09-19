import logging
import sys

import pandas as pd

# initialize logger
log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logging.basicConfig()

# Output of make_statistics.py should be fed in
EXPECTED_COLUMNS = [
    "treeoflife_id",
    "eol_content_id",
    "eol_page_id",
    "bioscan_part",
    "bioscan_filename",
    "inat21_filename",
    "inat21_cls_name",
    "inat21_cls_num",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "common",
]

TAXA = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def check_kingdom(df):
    """
    This function checks the number of distinct kingdoms labeled in the dataset (should be 3: Animalia, Plantae, and Fungi).
    Logs a warning with the number included if it's not 3. Warning is printed to terminal, not saved to file.

    Parameters:
    -----------
    df - DataFrame with taxonomic hierarchy as columns.

    """
    # check we have only 3 kingdoms
    num_kingdoms = df.kingdom.nunique()
    if num_kingdoms != 3:
        logging.warning(f"There are {num_kingdoms} kingdoms instead of 3.")


def check_hierarchy(df):
    """
    Function to check for gaps in the taxonomic hierarchy above the lowest known rank (eg., no missing 'family' if 'genus' specified).
    Logs gaps as warnings. These are printed to terminal, not saved to file.

    Parameters:
    -----------
    df - DataFrame with taxonomic hierarchy as columns.

    """
    # Want to check hierarchy is backfilled from genus up (if only species labeled, requires genus to diambiguate)
    # and similarly for at least family and order since those are lowest ranks for most BIOSCAN
    temp = df.loc[df.genus.notna()]
    for taxon in TAXA[:-2]:
        if temp[taxon].isna().sum() > 0:
            logging.warning(
                f"{temp[taxon].isna().sum()} entries are missing rank {taxon}, but have genus label."
            )

    # check family and above, but don't duplicate effort
    missing_genera = df.loc[df.genus.isna()]
    temp_fam = missing_genera.loc[missing_genera.family.notna()]
    for taxon in TAXA[:-3]:
        if temp_fam[taxon].isna().sum() > 0:
            logging.warning(
                f"{temp_fam[taxon].isna().sum()} entries are missing rank {taxon}, but have family label."
            )

    # check order and above, but don't duplicate effort
    missing_family = missing_genera.loc[missing_genera.family.isna()]
    temp_order = missing_family.loc[missing_family.order.notna()]
    for taxon in TAXA[:-4]:
        if temp_order[taxon].isna().sum() > 0:
            logging.warning(
                f"{temp_order[taxon].isna().sum()} entries are missing rank {taxon}, but have order label."
            )

    # check if higher order taxa are filled in for non-null species missing genus
    temp_species = missing_genera.loc[missing_genera.species.notna()]
    num = temp_species.shape[0]
    for taxon in TAXA[:-2]:
        if temp_species[taxon].isna().sum() < num:
            logging.warning(
                f"{num - temp_species[taxon].isna().sum()} entries have {taxon} and species labels but no genus."
            )


def check_sci_name(df):
    """
    This function checks the number of words in the species column for each sample.
    Logs a warning with the number that have more than one word indicating the potential for both genus and species to be recorded.
    Warning is printed to terminal, not saved to file.

    Parameters:
    -----------
    df - DataFrame with taxonomic hierarchy as columns.

    """
    # Check for scientific name in species column (i.e., genus speices in species column, may correspond to missing genus)
    count = 0
    for species in list(df.loc[df["species"].notna(), "species"]):
        if len(species.split(" ")) > 1:
            count += 1
    if count > 0:
        logging.warning(
            f"There are {count} samples with species label longer than one word."
        )


def check_id(df):
    """
    This function checks the species column for `(unidentified)`, since this appears in EOL data which is regularly updated.
    Logs a warning with the number of entries where species is labeled `(unidentified)`.
    Warning is printed to terminal, not saved to file.

    Parameters:
    -----------
    df - DataFrame with taxonomic hierarchy as columns.
    """
    num_noID = len(df.loc[df.species == "(unidentified)"])
    if num_noID > 0:
        logging.warning(f"{num_noID} species are labeled as '(unidentified)'.")


def check_common(df):
    """
    This function checks the common column for nulls, since this column should be filled with either common name or lowest order taxa.
    Logs a warning with the number of entries where common is null.
    Warning is printed to terminal, not saved to file.

    Parameters:
    -----------
    df - DataFrame with data source and common as columns.
    """
    missing_common = df.loc[df.common.isna()]
    num_no_common = len(missing_common)
    if num_no_common > 0:
        logging.warning(f"{num_no_common} entries have null common. They are from {missing_common.data_source.unique()}.")

    # print the first and last 5 entries with missing common
    if num_no_common > 0:
        logging.warning(f"First 5 entries with missing common: {missing_common.head()}")
        logging.warning(f"Last 5 entries with missing common: {missing_common.tail()}")




def main():
    # check for file
    if len(sys.argv) == 1:
        sys.exit("Please provide a source CSV file.")
    else:
        try:
            df = pd.read_csv(sys.argv[1], low_memory=False)
        except Exception as e:
            sys.exit(e)

        # check for columns
        missing_cols = []
        for col in EXPECTED_COLUMNS:
            if col not in list(df.columns):
                missing_cols.append(col)
        if len(missing_cols) > 0:
            sys.exit(f"Source CSV does not have {missing_cols} columns.")

    # If split column included, remove "train_small" entries as they are duplicates from "train".
    if "split" in list(df.columns):
        df = df.loc[df.split != "train_small"]

    # Add data_source column to pin-point missing info 
    df.loc[df["inat21_filename"].notna(), "data_source"] = "iNat21"
    df.loc[df["bioscan_filename"].notna(), "data_source"] = "BIOSCAN"
    df.loc[df["eol_content_id"].notna(), "data_source"] = "EOL"

    # Run checks
    # only pass taxa columns for efficiency (source and common for final check)
    check_kingdom(df[TAXA])
    check_hierarchy(df[TAXA])
    check_sci_name(df[TAXA])
    check_id(df[TAXA])
    check_common(df[["data_source", "common"]])

if __name__ == "__main__":
    main()
