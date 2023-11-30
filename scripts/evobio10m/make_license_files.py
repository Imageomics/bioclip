"""
Make license files for the various datasets that make up TreeOfLife-10M.
"""
import argparse
import csv
import logging
import pathlib
import itertools

from tqdm import tqdm

from imageomics import disk, evobio10m

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("main")


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, help="Path to output directory.")
    parser.add_argument("--db", required=True, help="Path to mapping.sqlite")

    return parser


def manifest_filepath(i):
    return pathlib.Path(disk.eol_media_manifest_dir) / f"media_manifest_{i}.csv"


def get_tol10m_ids(eol_content_ids, eol_page_ids):
    assert len(eol_content_ids) == len(eol_page_ids)
    placeholders = ",".join(["?"] * len(eol_content_ids))
    stmt = f"SELECT evobio10m_id, content_id, page_id FROM eol WHERE content_id IN ({placeholders}) AND page_id IN ({placeholders})"

    results = {}
    for tol10m_id, content_id, page_id in db.execute(
        stmt, eol_content_ids + eol_page_ids
    ).fetchall():
        results[(content_id, page_id)] = tol10m_id
    return results


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def write_from_manifest(reader, writer):
    # 999 is the maximum number of placeholders in the sqlite3 version on OSC.
    # Since we look for both page ids and content ids, we need 999 // 2
    for batch in tqdm(batched(reader, 999 // 2)):
        eol_content_ids, eol_page_ids, *_ = zip(*batch)

        results = get_tol10m_ids(eol_content_ids, eol_page_ids)

        for eol_content_id, eol_page_id, _, _, license, owner in batch:
            eol_content_id = int(eol_content_id)
            eol_page_id = int(eol_page_id)

            tol10m_id = results.get((eol_content_id, eol_page_id))
            if not tol10m_id:
                continue
            writer.writerow([tol10m_id, eol_content_id, eol_page_id, license, owner])



if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    db = evobio10m.get_db(args.db)

    with open(pathlib.Path(args.outdir) / "eol_licenses.csv", "w") as licenses_fd:
        writer = csv.writer(licenses_fd)

        # Write headers.
        writer.writerow(
            ["treeoflife10m_id", "eol_content_id", "eol_page_id", "license", "owner"]
        )

        i = 1
        while manifest_filepath(i).is_file():
            logger.info("Reading from %s.", manifest_filepath(i))
            with open(manifest_filepath(i)) as manifest_fd:
                reader = csv.reader(manifest_fd)
                if i == 1:
                    next(reader)  # skip header rows
                write_from_manifest(reader, writer)

            i += 1
            logger.info("Finished with %s.", manifest_filepath(i))
        licenses_fd.flush()
