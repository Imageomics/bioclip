import dataclasses
import sqlite3


def get_outdir(tag):
    return f"/fs/ess/PAS2136/open_clip/data/evobio10m-{tag}"


schema = """
CREATE TABLE IF NOT EXISTS eol (
    content_id INT NOT NULL,
    page_id INT NOT NULL,
    evobio10m_id TEXT NOT NULL PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS inat21 (
    filename TEXT NOT NULL,
    cls_name TEXT NOT NULL,
    cls_num INT NOT NULL,
    evobio10m_id TEXT NOT NULL PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS bioscan (
    part INT NOT NULL,
    filename TEXT NOT NULL,
    evobio10m_id TEXT NOT NULL PRIMARY KEY
);

-- evobio10m_id is a foreign key for one of the three other tables.
CREATE TABLE IF NOT EXISTS split (
    evobio10m_id TEXT NOT NULL PRIMARY KEY,
    is_val INTEGER NOT NULL,
    is_train_small INTEGER NOT NULL
);

PRAGMA journal_mode=WAL;  -- write-ahead log
"""


def get_db(path):
    try:
        db = sqlite3.connect(path, timeout=120)
    except sqlite3.OperationalError as err:
        print(f"Could not connect to {path} ({err}).")
        raise

    db.execute("PRAGMA busy_timeout = 120000;")  # 120 second timeout
    db.commit()
    db.executescript(schema)
    db.commit()
    return db


def load_splits(db_path):
    db = get_db(db_path)
    train_stmt = "SELECT evobio10m_id FROM split WHERE is_val = 0;"
    train_small_stmt = (
        "SELECT evobio10m_id FROM split WHERE is_val = 0 AND is_train_small = 1;"
    )
    val_stmt = "SELECT evobio10m_id FROM split WHERE is_val = 1;"

    splits = {
        "train": {row[0] for row in db.execute(train_stmt).fetchall()},
        "val": {row[0] for row in db.execute(val_stmt).fetchall()},
        "train_small": {row[0] for row in db.execute(train_small_stmt).fetchall()},
    }
    db.close()
    return splits


@dataclasses.dataclass(frozen=True)
class DatasetId:
    eol_content_id: int = None
    eol_page_id: int = None
    bioscan_part: int = None
    bioscan_filename: str = None
    inat21_filename: str = None
    inat21_cls_name: str = None
    inat21_cls_num: int = None

    def __post_init__(self):
        assert (
            (self.eol_content_id and self.eol_page_id)
            or (self.bioscan_part and self.bioscan_filename)
            or (
                self.inat21_filename
                and self.inat21_cls_name
                and self.inat21_cls_num is not None
            )
        ), repr(self)

        if self.inat21_filename:
            assert self.inat21_filename.endswith(".jpg")

        if self.bioscan_filename:
            assert self.bioscan_filename.endswith(".jpg")

    def to_tuple(self):
        return (
            self.eol_content_id,
            self.eol_page_id,
            self.bioscan_part,
            self.bioscan_filename,
            self.inat21_filename,
            self.inat21_cls_name,
            self.inat21_cls_num,
        )
