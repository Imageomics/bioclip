import pandas as pd
from pathlib import Path
import argparse
import sys

# Output of make_statistics.py (catalog.csv) should be fed in,
# will also work with predicted-catalog.csv.
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

SOURCE_TAXA = [
    "data_source",
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

DATA_SOURCES = ["TreeOfLife-10M", "iNat21", "BIOSCAN", "EOL"]


def get_taxa_info(df, stats_path):
    """
    Function to generate taxonomic statistics and save to CSV at desired location.
    Calculates total number of unique 7-tuples in full dataset and each constituent data source.

    Parameters:
    -----------
    df - DataFrame with data source, taxonomic hierarchy, and common as columns.
    stats_path - String. Path to location in which to save the stats file.
    """
    print("Fetching taxonomic stats")
    unique_taxa_counts = []
    # Get number of unique 7-tuples in full dataset and each constituent data source.
    for data_source in DATA_SOURCES:
        df_temp = df.copy()
        if data_source != "TreeOfLife-10M":
            df_temp = df_temp.loc[df_temp.data_source == data_source]
        df_temp["duplicate"] = df_temp.duplicated(subset=TAXA, keep="first")
        df_unique_taxa = df_temp.loc[~df_temp["duplicate"]]
        num_unique_taxa = df_unique_taxa.shape[0]
        unique_taxa_counts.append(num_unique_taxa)

    # Make DataFrame with these taxa stats
    taxa_counts_df = pd.DataFrame(
        data={"data_source": DATA_SOURCES, "num_unique_taxa": unique_taxa_counts}
    )

    # Save to CSV
    taxa_counts_df.to_csv(stats_path, index=False)


def main(src_csv, dest_dir):
    # Check CSV compatibility
    try:
        print("Reading your file")
        df = pd.read_csv(src_csv, low_memory=False)
    except Exception as e:
        sys.exit(e)

    # Check for columns
    missing_cols = []
    for col in EXPECTED_COLUMNS:
        if col not in list(df.columns):
            missing_cols.append(col)
    if len(missing_cols) > 0:
        sys.exit(f"Source CSV does not have {missing_cols} columns.")

    print("Processing your file")
    # If split column included, remove "train_small" entries as they are duplicates from "train".
    if "split" in list(df.columns):
        df = df.loc[df.split != "train_small"]

    # Add "data_source" column to pin-point missing info.
    df.loc[df["inat21_filename"].notna(), "data_source"] = "iNat21"
    df.loc[df["bioscan_filename"].notna(), "data_source"] = "BIOSCAN"
    df.loc[df["eol_content_id"].notna(), "data_source"] = "EOL"

    # Check filepath given by user
    dest_dir_path = Path(dest_dir)
    if not dest_dir_path.is_absolute():
        # Use bioclip as reference folder
        base_path = Path(__file__).parent.parent.parent.resolve()
        stats_path = base_path / dest_dir_path
    else:
        stats_path = dest_dir_path

    stats_path.mkdir(parents=True, exist_ok=True)

    # Reduce DataFrame for easier processing ("data_source", taxa, and "common" columns).
    get_taxa_info(df[SOURCE_TAXA], str(stats_path / "taxa_stats.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate taxonomic statistics file")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument(
        "--output_path",
        default="data/stats",
        required=False,
        help="Path to the folder for output stats file",
    )
    args = parser.parse_args()

    main(args.input_file, args.output_path)
