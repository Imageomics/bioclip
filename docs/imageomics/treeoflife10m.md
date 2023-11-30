# Dataset Card for TreeOfLife-10M


## How to Create TreeOfLife-10M

This is the process for creating the entire dataset, version 3.3 (which we used to train BioCLIP for the public release).

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
9. **[`check_taxa`](/scripts/evobio10m/check_taxa.py)**:
   - This will check the actual catalog file for any taxa issues.
   - More information on this file can be found [here](/scripts/README.md).
   - Run `python scripts/evobio10m/check_taxa.py /fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/catalog.csv`


This process is buggy and doesn't always work.
`make_wds.py` tries to re-write wds files that are corrupted, but it doesn't always work.
`make_wds.py` also ignores images and species used in the rare species benchmark.
