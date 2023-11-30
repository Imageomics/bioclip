import argparse
import csv
import logging
import os
import tarfile

import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm

from imageomics import disk, evobio10m, naming

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger()

headers = (
    "split",
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
)


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory with split directories, each which should contain .tar shard files.",
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of worker processes"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--tag", required=True, help="Tag for this run.")
    parser.add_argument("--db", required=True, help="Path to mapping.sqlite")

    return parser


def log_and_continue(err):
    if isinstance(err, tarfile.ReadError) and len(err.args) == 3:
        logger.warn(err.args[2])
        return True

    if isinstance(err, ValueError):
        return True

    raise err


def human(num):
    levels = [("G", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)]
    prefix = "-" if num < 0 else ""
    num = abs(num)
    for suffix, val in levels:
        if num >= val:
            return f"{prefix}{num / val:.1f}{suffix}"

    if num < 1:
        return f"{prefix}{num:.2g}"

    return f"{prefix}{num}"


def write_rows(split_dir, writer):
    shardlist = [
        os.path.join(split_dir, shard)
        for shard in os.listdir(split_dir)
        if shard.endswith(".tar")
    ]

    dataset = wds.DataPipeline(
        wds.SimpleShardList(shardlist),
        # Without this line, you will get num_workers copies
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.decode("torchrgb"),
        wds.to_tuple(*keys, handler=log_and_continue),
    )
    dataloader = DataLoader(
        dataset, num_workers=args.workers, batch_size=args.batch_size
    )
    seen = set()
    for batch in tqdm(dataloader):
        for key, common in zip(*batch):
            assert key not in seen
            seen.add(key)

            data_id = key_lookup[key]
            if data_id.eol_page_id:
                taxon, _ = eol_name_lookup[data_id.eol_page_id]
            elif data_id.bioscan_filename:
                taxon, _ = bioscan_name_lookup[key]
            elif data_id.inat21_cls_name:
                taxon, _ = inat21_name_lookup[data_id.inat21_cls_num]
            else:
                raise ValueError(data_id)

            writer.writerow(
                (split, key, *data_id.to_tuple(), *taxon.to_tuple(), common)
            )


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    db = evobio10m.get_db(args.db)
    logger.info("Started.")

    eol_name_lookup = naming.load_name_lookup(disk.eol_name_lookup_json, keytype=int)
    inat21_name_lookup = naming.load_name_lookup(
        disk.inat21_name_lookup_json, keytype=int
    )
    bioscan_name_lookup = naming.load_name_lookup(disk.bioscan_name_lookup_json)
    logger.info("Loaded name lookups.")

    key_lookup = {}
    # EOL
    stmt = "SELECT content_id, page_id, evobio10m_id FROM eol"
    for content, page, evobio10m_id in db.execute(stmt).fetchall():
        key_lookup[evobio10m_id] = evobio10m.DatasetId(
            eol_content_id=content, eol_page_id=page
        )

    # Bioscan
    stmt = "SELECT part, filename, evobio10m_id FROM bioscan"
    for part, filename, evobio10m_id in db.execute(stmt).fetchall():
        key_lookup[evobio10m_id] = evobio10m.DatasetId(
            bioscan_part=part, bioscan_filename=filename
        )

    # iNat21
    stmt = "SELECT filename, cls_name, cls_num, evobio10m_id FROM inat21"
    for filename, cls_name, cls_num, evobio10m_id in db.execute(stmt).fetchall():
        key_lookup[evobio10m_id] = evobio10m.DatasetId(
            inat21_filename=filename,
            inat21_cls_name=cls_name,
            inat21_cls_num=cls_num,
        )

    logger.info("Loaded keys.")

    keys = ("__key__", "common_name.txt")

    outfile = os.path.join(args.dir, "catalog.csv")
    with open(outfile, "w", newline="") as fd:
        writer = csv.writer(fd)
        writer.writerow(headers)

        for split in os.listdir(args.dir):
            split_dir = os.path.join(args.dir, split)
            if os.path.isfile(split_dir):
                logger.info(
                    "Skipping file because it's not a directory. [file: %s]", split_dir
                )
                continue

            write_rows(split_dir, writer)
            logger.info("Wrote split. [split: %s, file: %s]", split, outfile)
