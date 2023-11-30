# Imageomics-Specific Scripts

`inat21_to_wds.py` converts raw iNat21 images to a webdataset format and holds 1K classes out for an unseen evaluation. To use it, you will have to edit some variables in the script (`inat_root`, `output_root`). **Note: this script is outdated because we use a combination of scripts in `scripts/evobio10m/`, described in [training-data-osc.md](/docs/imageomics/training-data-osc.md)**

`inat_common_names.py` gets a mapping of scientific names to common names for all species in the iNat21 dataset. In the future, this will be expanded to all species on iNaturalist, not just those in the iNat21 splits. You don't need to run this script because the mapping is already committed to version control.

`evobio10m/check_taxa.py` reads in the full dataset CSV from [Step 6](/docs/imageomics/treeoflife10m.md) and checks for gaps in the taxonomic hierarchy:
```bash
python scripts/evobio10m/check_taxa.py <path/to/CSV>
```
The only requirement is `pandas`.
It prints warnings indicating the extent of gaps when they occur. A [test CSV](https://huggingface.co/datasets/imageomics/ToL-EDA/blob/main/data/tol_hierarchy_test.csv) for this script can be found on the [ToL-EDA HF repo](https://huggingface.co/datasets/imageomics/ToL-EDA). 

`evobio10m/taxa_viz.py` reads in the full dataset CSV from [Step 6](/docs/imageomics/treeoflife10m.md) and generates treemaps from kingdom and phylum down to family rank. Saves treemaps as both `HTML` (interactive) and `PDF` (static to print or display); defaults to `data/visuals` folder if no path given for visuals.
```bash
python scripts/evobio10m/taxa_viz.py --output_path <path/for/visuals> <path/to/CSV>
```
Requirements given in `requirements-viz.txt`. The test CSV for the `check_taxa` script can be used. The only requirements for the CSV are that it contain the taxa `kingdom` through `family`. If a deeper tree is desired, just add more taxa to the `TAXA` variable at the top of the script; `family` was set as the base due to the size of the dataset.

`evobio10m/get_data_stats.py` reads in the full dataset CSV from [Step 6](/docs/imageomics/treeoflife10m.md) and calculates the total number of unique 7-tuples (kingdom through species) in the full dataset and in each constituent data source (EOL, iNat21, and BIOSCAN-1M). This information is saved to `data/stats` folder by default if no path is given.
```bash
python scripts/evobio10m/get_data_stats.py --output_path <path/for/stats> <path/to/CSV>
```
The only requirement is `pandas`.
The test CSV for the `check_taxa` script can be used.
