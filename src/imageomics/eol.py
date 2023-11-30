import dataclasses
import logging
import re

from . import disk, helpers, naming

logger = logging.getLogger()

eol_filename_pattern = re.compile(r"(\d+)_(\d+)_eol.*jpg")


@dataclasses.dataclass(frozen=True)
class ImageFilename:
    """
    Represents a filename like 12784812_51655800_eol-full-size-copy.jpg
    """

    content_id: int
    page_id: int
    ext: str
    raw: str

    @classmethod
    def from_filename(cls, filename):
        match = eol_filename_pattern.match(filename)
        if not match:
            raise ValueError(filename)
        return cls(int(match.group(1)), int(match.group(2)), "jpg", filename)


class EolNameLookup:
    def __init__(self):
        # dict[page id, (rank, name, parent id)]
        # These are known errors that we preemptively correct.
        self._lookup = {
            331434: ("species", "latouchii", 42333),
            52338888: ("species", "lateralis", 42333),
        }

        self._resourcepk_to_pageid = {}  # dict[resource pk, page id]

        for row in helpers.csvreader(disk.eol_taxon_tab, delimiter="\t"):
            if "eolID" in row and row["eolID"]:
                page_id = int(row["eolID"])
            else:
                continue
            self._resourcepk_to_pageid[row["taxonID"]] = page_id

        # Takes about 21 seconds
        for row in helpers.csvreader(disk.eol_taxon_tab, delimiter="\t"):
            if "eolID" in row and row["eolID"]:
                page_id = int(row["eolID"])
            else:
                continue

            rank = row.get("taxonRank", "").lower()

            if rank == "class":
                rank = "cls"

            parent_resource_pk = row.get("parentNameUsageID", "")
            parent_page_id = self._resourcepk_to_pageid.get(parent_resource_pk)
            name = row["canonicalName"]

            if page_id in self._lookup:
                logger.info(
                    "Skipping page id because it's present. [page id: %s]", page_id
                )
                continue

            if (
                rank == "species"
                and name.capitalize() == name
                and len(name.split()) == 2
            ):
                # We can likely assume that the first word is the genus.
                genus, species = name.split()
                self._lookup[page_id] = ("species", species.lower(), parent_page_id)
            else:
                self._lookup[page_id] = (rank, name.lower(), parent_page_id)

    def guess_value(self, name, genus_lookup):
        if len(name.split()) == 2:
            genus, species = name.split()
            genus_pageid = genus_lookup.get(genus)
            return ("species", species, genus_pageid)

        return ("species", name, None)

    def add_low_quality_names(self, page_ids_with_imgs):
        """
        Low quality names often do not include the entire taxonomic chain. So we only add these names for the page ids that actually matter, which are the ones with images.
        """
        # Sometimes we can find a species' genus by searching by name.
        # It doesn't always work well because species names can be reused for different genera
        genus_lookup = {
            name: page_id
            for page_id, (rank, name, _) in self._lookup.items()
            if rank == "genus"
        }
        n_genus = sum(1 for (rank, _, _) in self._lookup.values() if rank == "genus")
        n_unique_genus = len(genus_lookup)
        logger.info(
            "Only %d/%d genus are unique (%.1f)%%",
            n_unique_genus,
            n_genus,
            n_unique_genus / n_genus * 100,
        )

        for row in helpers.csvreader(disk.eol_scraped_page_ids_csv):
            page_id = int(row["page_id"])
            if page_id in self._lookup or page_id not in page_ids_with_imgs:
                continue

            name = naming.clean_name(row["scientific_name"])
            if name in genus_lookup:
                logger.debug(
                    "Name '%s' (https://eol.org/pages/%d) is a genus (https://eol.org/pages/%d).",
                    name,
                    page_id,
                    genus_lookup.get(name),
                )

            value = self.guess_value(name, genus_lookup)
            if value:
                self._lookup[page_id] = value

    def taxon(self, page_id):
        taxon = {rank: "" for rank in naming.taxon_ranks}
        while page_id in self._lookup:
            rank, value, parent_id = self._lookup[page_id]
            if rank in naming.taxon_ranks:
                taxon[rank] = value
            page_id = parent_id

        return naming.Taxon(**taxon)
