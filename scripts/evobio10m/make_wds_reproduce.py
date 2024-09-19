"""
Writes the training and validation data to webdataset format.
"""
import argparse
import collections
import json
import logging
import multiprocessing
import os
import tarfile

from PIL import Image, ImageFile

from imageomics import disk_reproduce, eol_reproduce, evobio10m_reproduce, naming_reproduce, wds

########
# CONFIG
########

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
rootlogger = logging.getLogger("root")

Image.MAX_IMAGE_PIXELS = 30_000**2  # 30_000 pixels per side
ImageFile.LOAD_TRUNCATED_IMAGES = True


########
# SHARED
########


def load_img(file):
    img = Image.open(file)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    return img.resize(resize_size, resample=Image.BICUBIC)


def load_blacklists():
    image_blacklist = set()
    species_blacklist = set()

    with open(disk_reproduce.seen_in_training_json) as fd:
        for scientific, images in json.load(fd).items():
            image_blacklist |= set(os.path.basename(img) for img in images)

    with open(disk_reproduce.unseen_in_training_json) as fd:
        for scientific, images in json.load(fd).items():
            image_blacklist |= set(os.path.basename(img) for img in images)
            species_blacklist.add(scientific)

    return image_blacklist, species_blacklist


######################
# Encyclopedia of Life
######################


def copy_eol_from_tar(sink, imgset_path):
    logger = logging.getLogger(f"p{os.getpid()}")

    db = evobio10m_reproduce.get_db(db_path)
    select_stmt = "SELECT evobio10m_id, content_id, page_id FROM eol;"
    eol_ids_lookup = {
        evobio10m_id: (content_id, page_id)
        for evobio10m_id, content_id, page_id in db.execute(select_stmt).fetchall()
    }
    db.close()

    name_lookup = naming_reproduce.load_name_lookup(disk_reproduce.eol_name_lookup_json, keytype=int)

    # r|gz indcates reading from a gzipped file, streaming only
    with tarfile.open(imgset_path, "r|gz") as tar:
        for i, member in enumerate(tar):
            eol_img = eol_reproduce.ImageFilename.from_filename(member.name)
            if eol_img.raw in image_blacklist:
                continue
            
            # Match on treeoflife_id filename
            if eol_img.tol_id not in eol_ids_lookup:
                print(f"Can't find the tol_id {eol_img.tol_id}")
                logger.warning(
                    "EvoBio10m ID missing. [tol_id: %s]",
                    eol_img.tol_id,
                )
                continue

            # fetching page ID
            content_id, page_id = eol_ids_lookup[eol_img.tol_id]
            global_id = eol_img.tol_id
            
            # checking for global id in split
            if global_id not in splits[args.split] or global_id in finished_ids:
                continue
            
            # checking for page id
            if page_id not in name_lookup:
                continue
            
            # using name lookup for taxon, common, classes
            taxon, common, classes = name_lookup[page_id]

            if taxon.scientific in species_blacklist:
                continue

            file = tar.extractfile(member)
            try:
                img = load_img(file).resize(resize_size)
            except OSError as err:
                logger.warning(
                    "Error opening file. Skipping. [tar: %s, err: %s]", imgset_path, err
                )
                continue

            txt_dct = make_txt(taxon, common) 
            # writing EOL
            sink.write(
                {"__key__": global_id, "jpg": img, **txt_dct} #, "classes": classes} # REMOVED "classes", unused list of numbers, throws an error as an unknown extension with webdataset
            )


########
# INAT21
########


def copy_inat21_from_clsdir(sink, clsdir):
    logger = logging.getLogger(f"p{os.getpid()}")

    db = evobio10m_reproduce.get_db(db_path)
    select_stmt = "SELECT evobio10m_id, filename, cls_num FROM inat21;"
    evobio10m_id_lookup = {
        (filename, cls_num): evobio10m_id
        for evobio10m_id, filename, cls_num in db.execute(select_stmt).fetchall()
    }
    db.close()

    name_lookup = naming_reproduce.load_name_lookup(disk_reproduce.inat21_name_lookup_json, keytype=int)

    clsdir_path = os.path.join(disk_reproduce.inat21_root_dir, clsdir)
    for i, filename in enumerate(os.listdir(clsdir_path)):
        filepath = os.path.join(clsdir_path, filename)

        cls_num, *_ = clsdir.split("_")
        cls_num = int(cls_num)

        if (filename, cls_num) not in evobio10m_id_lookup:
            logger.warning(
                "Evobio10m ID missing. [image: %s, cls: %d]", filename, cls_num
            )
            continue

        global_id = evobio10m_id_lookup[(filename, cls_num)]
        if global_id not in splits[args.split] or global_id in finished_ids:
            continue

        taxon, common, classes = name_lookup[cls_num]

        if taxon.scientific in species_blacklist:
            continue

        txt_dct = make_txt(taxon, common) #, classes)
        img = load_img(filepath).resize(resize_size)
        # writing iNat
        sink.write({"__key__": global_id, "jpg": img, **txt_dct} #, "classes": classes} # REMOVED "classes", unused list of numbers, throws an error as an unknown extension with webdataset
                   )


#########
# BIOSCAN
#########


def copy_bioscan_from_part(sink, part):
    logger = logging.getLogger(f"p{os.getpid()}")

    db = evobio10m_reproduce.get_db(db_path)
    select_stmt = "SELECT evobio10m_id, part, filename FROM bioscan;"
    evobio10m_id_lookup = {
        (part, filename): evobio10m_id
        for evobio10m_id, part, filename in db.execute(select_stmt).fetchall()
    }
    db.close()

    name_lookup = naming_reproduce.load_name_lookup(disk_reproduce.bioscan_name_lookup_json)

    partdir = os.path.join(disk_reproduce.bioscan_root_dir, f"part{part}")
    for i, filename in enumerate(os.listdir(partdir)):
        if (part, filename) not in evobio10m_id_lookup:
            logger.warning(
                "EvoBio10m ID missing. [part: %d, filename: %s]", part, filename
            )
            continue

        global_id = evobio10m_id_lookup[(part, filename)]
        if global_id not in splits[args.split] or global_id in finished_ids:
            continue

        taxon, common, classes = name_lookup[global_id]

        if taxon.scientific in species_blacklist:
            continue

        txt_dct = make_txt(taxon, common) #, classes)
        filepath = os.path.join(partdir, filename)
        img = load_img(filepath).resize(resize_size)
        # writing BIOSCAN
        sink.write({"__key__": global_id, "jpg": img, **txt_dct} #, "classes": classes} # REMOVED "classes", unused list of numbers, throws an error as an unknown extension with webdataset
                   )


######
# MAIN
######


def check_status():
    finished_ids, bad_shards = set(), set()
    for root, dirs, files in os.walk(outdir):
        for file in files:
            if not file.endswith(".tar"):
                continue

            written = collections.defaultdict(set)
            with tarfile.open(os.path.join(root, file), "r|") as tar:
                try:
                    for member in tar:
                        global_id, *rest = member.name.split(".")
                        rest = ".".join(rest)
                        written[global_id].add(rest)
                except tarfile.TarError as err:
                    print(err)
                    continue

            # If you change make_txt, update these expected_texts
            expected_exts = {
                "scientific_name.txt",
                "taxonomic_name.txt",
                "common_name.txt",
                "sci.txt",
                "com.txt",
                "taxon.txt",
                "taxonTag.txt",
                "sci_com.txt",
                "taxon_com.txt",
                "taxonTag_com.txt",
                "jpg",
            }
            for global_id, exts in written.items():
                if exts != expected_exts:
                    # Delete all the files with this global_id. But this is impossible
                    # with the .tar format. So instead, we delete this entire shard.
                    bad_shards.add(os.path.join(root, file))
                    break
            else:
                # If we didn't early break, then this file is clean.
                finished_ids.update(set(written.keys()))

    return finished_ids, bad_shards


def make_txt(taxon, common):
    assert taxon is not None
    assert not taxon.empty, f"{common} has no taxon!"

    # test replacing the two if statements with
    common = naming_reproduce.get_common(taxon, common)
    '''if not common:
        common = taxon.scientific
    if not common:
        common = taxon.taxonomic'''

    # ex: kingdom Animalia phylum Arthropoda class ...
    tagged = " ".join(f"{tier} {value}" for tier, value in taxon.tagged)

    # IF YOU UPDATE THE KEYS HERE, BE SURE TO UPDATE check_status() TO LOOK FOR ALL OF
    # THESE KEYS. IF YOU DO NOT, THEN check_status() MIGHT ASSUME AN EXAMPLE IS WRITTEN
    # EVEN IF NOT ALL KEYS ARE PRESENT.
    return {
        # Names
        "scientific_name.txt": taxon.scientific,
        "taxonomic_name.txt": taxon.taxonomic,
        "common_name.txt": common,
        # "A photo of"... captions
        "sci.txt": f"a photo of {taxon.scientific}.",
        "com.txt": f"a photo of {common}.",
        "taxon.txt": f"a photo of {taxon.taxonomic}.",
        "taxonTag.txt": f"a photo of {tagged}.",
        "sci_com.txt": f"a photo of {taxon.scientific} with common name {common}.",
        "taxon_com.txt": f"a photo of {taxon.taxonomic} with common name {common}.",
        "taxonTag_com.txt": f"a photo of {tagged} with common name {common}.",
    }


sentinel = "STOP"


def worker(input):
    logger = logging.getLogger(f"p{os.getpid()}")
    with wds.ShardWriter(outdir, shard_counter, digits=6, maxsize=3e9) as sink:
        for func, args in iter(input.get, sentinel):
            logger.info(f"Started {func.__name__}({', '.join(map(str, args))})")
            func(sink, *args)
            logger.info(f"Finished {func.__name__}({', '.join(map(str, args))})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--width", type=int, default=224, help="Width of resized images."
    )
    parser.add_argument(
        "--height", type=int, default=224, help="Height of resized images."
    )
    parser.add_argument(
        "--split", choices=["train", "val", "train_small"], default="val"
    )
    parser.add_argument("--tag", default="dev", help="The suffix for the directory.")
    parser.add_argument(
        "--workers", type=int, default=32, help="Number of processes to use."
    )
    args = parser.parse_args()

    # Set up some global variables that depend on CLI args.
    resize_size = (args.width, args.height)
    outdir = f"{evobio10m_reproduce.get_outdir(args.tag)}/{args.width}x{args.height}/{args.split}"
    os.makedirs(outdir, exist_ok=True)
    print(f"Writing images to {outdir}.")

    # db_path = f"{evobio10m_reproduce.get_outdir(args.tag)}/mapping.sqlite"
    db_path = os.path.abspath(disk_reproduce.db)

    # Load train/val/train_small splits
    splits = evobio10m_reproduce.load_splits(db_path)

    # Load images already written to tar files to avoid duplicate work.
    # Delete any unfinished shards.
    finished_ids, bad_shards = check_status()
    rootlogger.info("Found %d finished examples.", len(finished_ids))
    rootlogger.warning("Found %d bad shards.", len(bad_shards))
    if bad_shards:
        for shard in bad_shards:
            os.remove(shard)
            rootlogger.warning("Deleted shard %d", shard)

    # Load image and species blacklists for rare species
    image_blacklist, species_blacklist = load_blacklists()

    # Creates a shared integer
    shard_counter = multiprocessing.Value("I", 0, lock=True)

    # All jobs read from this queue
    task_queue = multiprocessing.Queue()

    # Submit all tasks
    # EOL
    for imgset_name in sorted(os.listdir(disk_reproduce.eol_root_dir)):
        assert imgset_name.endswith(".tar.gz")
        imgset_path = os.path.join(disk_reproduce.eol_root_dir, imgset_name)
        task_queue.put((copy_eol_from_tar, (imgset_path,)))

    # Bioscan
    # 113 parts in bioscan
    for i in range(1, 114):
        task_queue.put((copy_bioscan_from_part, (i,)))

    # iNat
    for clsdir in os.listdir(disk_reproduce.inat21_root_dir):
        task_queue.put((copy_inat21_from_clsdir, (clsdir,)))

    processes = []
    # Start worker processes
    for i in range(args.workers):
        p = multiprocessing.Process(target=worker, args=(task_queue,))
        processes.append(p)
        p.start()

    # Stop worker processes
    for i in range(args.workers):
        task_queue.put(sentinel)

    for p in processes:
        p.join()
