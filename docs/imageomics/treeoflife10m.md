# How to Create TreeOfLife-10M

**Note:** 
- [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M) has the EOL images, but not iNat21 or BIOSCAN-1M due to licensing restrictions. 
- To reconstruct the _full_ dataset, please follow the steps outlined below in _Reproduce TreeOfLife-10M_. This reproduction process is designed to be run on an HPC system using Slurm.

## Reproduce TreeOfLife-10M

All of the following steps should be completed in the root directory of the repository. Start by setting up your conda environment with [`requirements-training.yml`](/requirements-training.yml):

```
conda env create -f requirements-training.yml --solver=libmamba -y
conda activate bioclip-train
pip install -e .
```

1. **Download [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M)**:
   - _Optional:_ Change the dataset storage location and other Slurm parameters (within the "customize" section) in the component download setup script ([`scripts/setup_download_tol-10m_components.bash`](/scripts/setup_download_tol-10m_components.bash)).
   - Download [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M) components by running:
     ```bash
	  sbatch --account <HPC-account> scripts/submit_download_tol-10m_components.bash
     ```
     This will download the tar and metadata files from Hugging Face, as well as [iNat21](https://github.com/visipedia/inat_comp/tree/master/2021#data) and [BIOSCAN-1M](https://zenodo.org/doi/10.5281/zenodo.8030064) into `../data/TreeOfLife-10M/` relative to the script, in the format specified in [`disk_reproduce`](/src/imageomics/disk_reproduce.py).
     - Note: This launches a collection of scripts which can also be run individually.
2. **[`make-dataset-wds_reproduce`](/slurm/make-dataset-wds_reproduce.sh)**:
   - This actually creates the webdataset files by running [`make_wds_reproduce`](/scripts/evobio10m/make_wds_reproduce.py) for each of the splits.
   - Make appropriate adjustments for your local setup to [`make-dataset-wds_reproduce`](/slurm/make-dataset-wds_reproduce.sh) (i.e., change account and path information, settings as described below).
   - On your HPC, run:
     ```bash
	  sbatch --account <HPC-account> slurm/make-dataset-wds_reproduce.sh
     ```
      - This runs the `scripts/evobio10m/make_wds_reproduce.py` for each of the splits using 32 workers.
      - It takes a long time (6 hours) and requires lots of memory.
3. **[`check_wds`](/scripts/evobio10m/check_wds.py)**:
   - Checks for bad shards and records them.
   - Run
     ```bash
	  sbatch --account <HPC-account> --cpus-per-task <num-CPUs> slurm/check-wds.slurm <shards> 
     ``` 
       - Writes a list of bad shards to `logs/bad-shards.txt`.
       - For instance, if images are placed in the default location, run the following to check the training split:
     ```bash
	  sbatch --account <HPC-account> --cpus-per-task 32 slurm/check-wds.slurm 'data/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/train/shard-{000000..000165}.tar'
     ```   
4. **[`make_catalog_reproduce`](/scripts/evobio10m/make_catalog_reproduce.py)**:
   - Generates the catalog of all images in the dataset, which includes information about their original data source and taxonomic record.
   - Run
     ```bash
	  sbatch --account <HPC-account> --cpus-per-task <N> slurm/make-catalog_reproduce.slurm \
	  --dir <path/to/splits> \
	  --db <path/to/db> \
	  --tag <tag> \
	  --batch-size <batch-size>
     ```
       - Creates a file `catalog.csv` in `--dir` which is a list of all names in the webdataset.
       - **Note:** `mapping.sqlite` is a SQLite database comprised of just the `predicted-catalog.csv` and can be replaced by a SQLite database constructed from [TreeOfLife-10M/metadata/catalog.csv](https://huggingface.co/datasets/imageomics/TreeOfLife-10M/blob/main/metadata/catalog.csv), which may be overwritten on this step depending on where these are saved.
       - For instance, if images are placed in the default location, run the following to generate the catalog file:
     ```bash
	  sbatch --account <HPC-account> --cpus-per-task 32 slurm/make-catalog_reproduce.slurm \
	  --dir data/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224 \
	  --db data/TreeOfLife-10M/metadata/mapping.sqlite \
	  --tag CVPR-2024 \
	  --batch-size 256
     ```
5. **[`check_taxa`](/scripts/evobio10m/check_taxa.py)**:
   - This will check the actual catalog file for any taxa issues.
   - More information on this file can be found [here](/scripts/README.md).
   - Run
     ```bash
	  python scripts/evobio10m/check_taxa.py /<path-to>/data/evobio10m-CVPR-2024/catalog.csv
     ```


## Original TreeOfLife-10M Generation
This was the process for creating the entire dataset, version 3.3 (which we used to train [BioCLIP](https://huggingface.co/imageomics/bioclip) for the public release).

1. **[`download_data`](/scripts/download_data.sh)**:
   - Run `bash scripts/download_data.sh` to download most of the metadata files.
2. **[`make_mapping`](/scripts/evobio10m/make_mapping.py)**:
   - Creates the sqlite database that maps from original files to tree of life ids.
   - Run `python scripts/evobio10m/make_mapping.py --tag v3.3 --workers 8`
     - Can run on login nodes and should take several hours. If you want it much faster, you can queue it on slurm with more workers.
3. **[`make_splits`](/scripts/evobio10m/make_splits.py)**:
   - Adds the splits table to the sqlite database: marks each image as belonging to either val or train, and then picks out 10% of the training images to use as an ablation study.
   - Run `python scripts/evobio10m/make_splits.py --db /fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/mapping.sqlite --val-split 5 --train-small-split 10 --seed 17`
       - This will run quickly on a login node.
4. **[`make_metadata`](/scripts/evobio10m/make_metadata.py)**:
   - Creates all the metadata files that can be easily used by `make_wds.py`. 
   - Also makes a `predicted-catalog.csv` file that will closely mimic `catalog.csv` (described below). `predicted-catalog.csv` includes rows for the rare species which are not included in `catalog.csv`.
       - See [ToL-EDA HF Repo](https://huggingface.co/datasets/imageomics/ToL-EDA) for more information about these files.
   - Run `python scripts/evobio10m/make_metadata.py --db /fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/mapping.sqlite` 
5. **[`check_taxa`](/scripts/evobio10m/check_taxa.py)**:
   - This will check the predicted catalog file for any taxa issues. If there are major issues, fix them first.
   - Run `python scripts/evobio10m/check_taxa.py /fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/predicted-catalog.csv` 
6. **[`make-dataset-wds`](/slurm/make-dataset-wds.sh)**:
   - This actually creates the webdataset files by running [`make_wds`](/scripts/evobio10m/make_wds.py) for each of the splits.
   - Run `sbatch slurm/make-dataset-wds.sh` on Pitzer.
      - This runs the `scripts/evobio10m/make_wds.py` for each of the splits using 32 workers.
      - It takes a long time (6 hours) and requires lots of memory.
7. **[`check_wds`](/scripts/evobio10m/check_wds.py)**:
   - Checks for bad shards and records them.
   - Run `scripts/evobio10m/check_wds.py --shardlist SHARDS --workers 8 > logs/bad-shards.txt` 
       - Writes a list of bad shards to `logs/bad-shards.txt`.
8. **[`make_catalog`](/scripts/evobio10m/make_catalog.py)**:
   - Generates the catalog of all images in the dataset, which includes information about their original data source and taxonomic record.
   - Run `python scripts/evobio10m/make_catalog.py --dir /fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/224x224/ --workers 8 --batch-size 256 --tag v3.3 --db /fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/mapping.sqlite`
       - Creates a file `catalog.csv` in `--dir` which is a list of all names in the webdataset.
       - **Note:** `mapping.sqlite` is a SQLite database comprised of just the `predicted-catalog.csv` and can be replaced by a SQLite database constructed from [TreeOfLife-10M/metadata/catalog.csv](https://huggingface.co/datasets/imageomics/TreeOfLife-10M/blob/main/metadata/catalog.csv), which may be overwritten on this step depending on where these are saved.
9. **[`check_taxa`](/scripts/evobio10m/check_taxa.py)**:
   - This will check the actual catalog file for any taxa issues.
   - More information on this file can be found [here](/scripts/README.md).
   - Run `python scripts/evobio10m/check_taxa.py /fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/catalog.csv`


This process is buggy and doesn't always work.
`make_wds.py` tries to re-write wds files that are corrupted, but it doesn't always work.
`make_wds.py` also ignores images and species used in the rare species benchmark.
