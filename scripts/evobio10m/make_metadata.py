"""
Parses all metadata files into a single format that's easily read by make_wds.py.
"""
import argparse
import csv
import json
import logging
import os

from tqdm import tqdm

from imageomics import disk, eol, evobio10m, naming

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger()


class ClassIndex:
    """
    Keeps track of ids for each tier and for the total taxon.
    """

    def __init__(self):
        self.tiered_lookups = [{} for _ in range(7)]

    def get(self, taxon, *, raise_if_missing=False):
        taxon = taxon.to_tuple()

        ids = []
        name = ""
        for i, value in enumerate(taxon):
            name += value
            if name not in self.tiered_lookups[i]:
                if raise_if_missing:
                    raise KeyError(f"Name {name} not in index[{i}]")
                else:
                    self.tiered_lookups[i][name] = len(self.tiered_lookups[i])
            ids.append(self.tiered_lookups[i][name])

        return tuple(ids)


def dump_json(obj, path):
    with open(path, "w") as fd:
        json.dump(obj, fd)
    logger.info("Dumped json object. [file %s]", path)


def make_bioscan_lookup():
    bioscan_name_lookup = naming.BioscanNameLookup()

    lookup = {}
    stmt = "SELECT evobio10m_id, filename FROM bioscan;"
    for global_id, filename in tqdm(db.execute(stmt).fetchall(), desc="Bioscan"):
        taxon = bioscan_name_lookup.taxon(filename)
        taxon = upgrader.upgrade(taxon)
        common = common_name_lookup.get(taxon.scientific, "")
        lookup[global_id] = (taxon.to_tuple(), common, class_index.get(taxon))

    dump_json(lookup, disk.bioscan_name_lookup_json)


def make_inat21_lookup():
    inat21_name_lookup = naming.iNat21NameLookup()

    lookup = {}
    stmt = "SELECT DISTINCT(cls_num) FROM inat21;"
    for (cls_num,) in tqdm(db.execute(stmt).fetchall(), desc="iNat21"):
        taxon = inat21_name_lookup.taxon(cls_num)
        common = common_name_lookup.get(taxon.scientific, "")
        lookup[cls_num] = (taxon.to_tuple(), common, class_index.get(taxon))

    dump_json(lookup, disk.inat21_name_lookup_json)


def make_eol_lookup():
    stmt = "SELECT distinct(page_id) FROM eol;"
    page_ids = set(id for (id,) in tqdm(db.execute(stmt).fetchall(), desc="page ids"))

    eol_name_lookup = eol.EolNameLookup()
    eol_name_lookup.add_low_quality_names(page_ids)

    lookup = {}

    for page_id in tqdm(page_ids, desc="images"):
        taxon = eol_name_lookup.taxon(page_id)
        if taxon.empty:
            continue
        taxon = upgrader.upgrade(taxon)
        common = common_name_lookup.get(taxon.scientific, "")
        lookup[page_id] = (taxon.to_tuple(), common, class_index.get(taxon))

    dump_json(lookup, disk.eol_name_lookup_json)


def make_predicted_catalog_csv(outfile):
    """
    Makes a predicted-catalog.csv file that's as close as possible to the other catalog.csv that is produced *after* making the webdataset.

    This doesn't ignore images that are in the species blacklist, so predicted-catalog.csv will have more rows than catalog.csv.
    """
    # Copied from make_catalog_csv in make_wds.py
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
        "class_index",
    )

    splits = evobio10m.load_splits(args.db)
    splits.pop("train_small")

    def get_split():
        return next(split for split, ids in splits.items() if global_id in ids)

    with open(outfile, "w", newline="") as fd:
        writer = csv.writer(fd)
        writer.writerow(headers)

        # EOL
        name_lookup = naming.load_name_lookup(disk.eol_name_lookup_json, keytype=int)
        stmt = "SELECT evobio10m_id, content_id, page_id FROM eol;"
        for global_id, content_id, page_id in db.execute(stmt).fetchall():
            data_id = evobio10m.DatasetId(
                eol_content_id=content_id, eol_page_id=page_id
            )
            if page_id not in name_lookup:
                continue
            taxon, common, cls_idx = name_lookup[page_id]
            common = naming.get_common(taxon, common)

            writer.writerow(
                (
                    get_split(),
                    global_id,
                    *data_id.to_tuple(),
                    *taxon.to_tuple(),
                    common,
                    cls_idx,
                )
            )
        logger.info("Wrote EOL rows. [file %s]", outfile)

        # Bioscan
        name_lookup = naming.load_name_lookup(disk.bioscan_name_lookup_json)
        stmt = "SELECT evobio10m_id, part, filename FROM bioscan"
        for global_id, part, filename in db.execute(stmt).fetchall():
            data_id = evobio10m.DatasetId(bioscan_part=part, bioscan_filename=filename)
            taxon, common, cls_idx = name_lookup[global_id]
            common = naming.get_common(taxon, common)
            writer.writerow(
                (
                    get_split(),
                    global_id,
                    *data_id.to_tuple(),
                    *taxon.to_tuple(),
                    common,
                    cls_idx,
                )
            )
        logger.info("Wrote Bioscan rows. [file %s]", outfile)

        # iNat21
        name_lookup = naming.load_name_lookup(disk.inat21_name_lookup_json, keytype=int)
        stmt = "SELECT evobio10m_id, filename, cls_name, cls_num FROM inat21"
        for global_id, filename, cls_name, cls_num in db.execute(stmt).fetchall():
            data_id = evobio10m.DatasetId(
                inat21_filename=filename,
                inat21_cls_name=cls_name,
                inat21_cls_num=cls_num,
            )
            taxon, common, cls_idx = name_lookup[cls_num]
            common = naming.get_common(taxon, common)
            writer.writerow(
                (
                    get_split(),
                    global_id,
                    *data_id.to_tuple(),
                    *taxon.to_tuple(),
                    common,
                    cls_idx,
                )
            )
        logger.info("Wrote iNat21 rows. [file %s]", outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to mapping.sqlite")
    args = parser.parse_args()

    assert os.path.isfile(args.db)
    db = evobio10m.get_db(args.db)
    logger.info("Connected to db. [db: %s]", args.db)

    common_name_lookup = naming.CommonNameLookup()
    dump_json(common_name_lookup, disk.common_name_lookup_json)

    upgrader = naming.NameUpgrader()

    class_index = ClassIndex()

    make_bioscan_lookup()
    make_inat21_lookup()
    make_eol_lookup()

    make_predicted_catalog_csv(
        os.path.join(os.path.dirname(args.db), "predicted-catalog.csv")
    )
