import pandas as pd
import plotly.express as px
from pathlib import Path
import argparse
import sys

# Output of make_statistics.py (catalog.csv) should be fed in,
# will also work with predicted-catalog.csv or a subset with just the taxa.
# TAXA are required columns (hierarchy to include in tree).
TAXA = ["kingdom", "phylum", "class", "order", "family"]


def get_colors(df, top_level):
    """
    Function to define color sequence to use for treemap.

    Parameters:
    -----------
    df - DataFrame with taxonomic hierarchy as columns.
    top_level - String. Level at which to start the tree ("kingdom" or "phylum").

    Returns:
    --------
    color_map - Dictionary. Mapping of kingdoms/phyla to color from Plotly Bold color scheme.
    """
    # Get list of all possible taxa for the level
    top_level_options = list(df[top_level].unique())
    # Generate color mapping for all kingdoms/phyla
    colors = px.colors.qualitative.Bold
    color_map = {}
    i = 0
    for option in top_level_options:
        # There are only 10 colors in the sequence, so we'll need to loop through it to assign all kingdoms/phyla
        i = i % 10
        color_map[option] = colors[i]
        i += 1
    return color_map


def make_tree(df, color_map, top_level, path_idx):
    """
    Function to generate a treemap from the designated top_level rank down to "family".

    Parameters:
    -----------
    df - DataFrame with taxonomic hierarchy as columns.
    color_map - Dictionary. Mapping of kingdoms/phyla to color from Plotly Bold color scheme.
    top_level - String. Level at which to start the tree ("kingdom" or "phylum").
    path_idx - Int. Index at which to start for TAXA (0 or 1).

    Returns:
    --------
    fig - Treemap figure from the designated top_level rank down to "family".
    """
    df_rank = df.copy()
    # Distribution of kingdom or phyla and lower taxa (to family) in TreeOfLife10M
    fig = px.treemap(
        df_rank, path=TAXA[path_idx:], color=top_level, color_discrete_map=color_map
    )

    # Minimize margins, set aspect ratio to 2:1
    fig.update_scenes(aspectratio={"x": 2, "y": 1})
    fig.update_layout(font={"size": 18}, margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return fig


def get_tree(df, top_level):
    """
    Function to filter DataFrame for designated top_level taxa, then get associated color mapping and treemap figure.

    Parameters:
    -----------
    df - DataFrame with taxonomic hierarchy as columns.
    top_level - String. Level at which to start the tree ("kingdom" or "phylum").

    Returns:
    --------
    fig - Treemap figure from the designated top_level rank down to "family".
    """
    print(f"Making {top_level} treemap")
    df_taxa = df.copy()
    if top_level == "phylum":
        # Drop null phylum values
        df_taxa = df_taxa.loc[df_taxa.phylum.notna()]
        # Set index for path
        path_idx = 1
    else:
        # Drop null kingdom values
        df_taxa = df_taxa.loc[df_taxa.kingdom.notna()]
        # Set index for path
        path_idx = 0
    # Fill null lower ranks with "unknown" for graphing purposes
    df_taxa = df_taxa.fillna("unknown")

    # Get color mapping for chosen top taxon and generate the treemap.
    color_map = get_colors(df_taxa, top_level)
    fig = make_tree(df_taxa, color_map, top_level, path_idx)
    return fig


def save_trees(df, viz_path):
    """
    Function to get and save treemap figures starting at both kingdom and phylum level.
    Saves as interactive HTML files and PDFs in the designated location.

    Parameters:
    -----------
    df - DataFrame with taxonomic hierarchy as columns.
    viz_path - Path object. Path to location in which to save the figures.

    """
    # Make phylum tree
    fig_phyla = get_tree(df, "phylum")
    # Make kingdom tree
    fig_kingdom = get_tree(df, "kingdom")

    # Save HTML of trees for interactive use
    phyla_html_path = str(viz_path / "phyla_ToL_tree.html")
    king_html_path = str(viz_path / "kingdom_ToL_tree.html")
    fig_phyla.write_html(phyla_html_path)
    fig_kingdom.write_html(king_html_path)

    # Save PDF of trees for publication
    # Aspect ratio set in the plot doesn't work for export (unless using the png export on the graph itself), so set the size manually.
    phyla_html_path = str(viz_path / "phyla_ToL_tree.pdf")
    king_html_path = str(viz_path / "kingdom_ToL_tree.pdf")
    fig_phyla.write_image(phyla_html_path, width=900, height=450)
    fig_kingdom.write_image(king_html_path, width=900, height=450)


def main(src_csv, dest_dir):
    # Check CSV compatibility
    try:
        print("Reading CSV")
        df = pd.read_csv(src_csv, low_memory=False)
    except Exception as e:
        sys.exit(e)

    # Check for columns
    print("Processing data")
    missing_cols = []
    for col in TAXA:
        if col not in list(df.columns):
            missing_cols.append(col)
    if len(missing_cols) > 0:
        sys.exit(f"Source CSV does not have {missing_cols} columns.")

    # If split column included, remove "train_small" entries as they are duplicates from "train".
    if "split" in list(df.columns):
        df = df.loc[df.split != "train_small"]

    # Check filepath given by user
    dest_dir_path = Path(dest_dir)
    if not dest_dir_path.is_absolute():
        # Use bioclip as reference folder
        base_path = Path(__file__).parent.parent.parent.resolve()
        viz_path = base_path / dest_dir_path
    else:
        viz_path = dest_dir_path

    viz_path.mkdir(parents=True, exist_ok=True)

    # Make and save trees to chosen filepath
    save_trees(df[TAXA], viz_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make taxonomic hierarchy visualizations"
    )
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument(
        "--output_path",
        default="data/visuals",
        required=False,
        help="Path to the folder for output visualization files",
    )
    args = parser.parse_args()

    main(args.input_file, args.output_path)
