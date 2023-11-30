import collections
import dataclasses
import json
import logging
import os
import re

from tqdm import tqdm

from . import disk, helpers

logger = logging.getLogger()

taxon_ranks = ("kingdom", "phylum", "cls", "order", "family", "genus", "species")


def clean_rank(value):
    assert isinstance(value, str)
    # Remove HTML
    value = strip_html(value).lower()

    # Sometimes values are not known, but not empty.
    if value in ("(unidentified)", "unknown", "pleasehelp"):
        value = ""
    return value


@dataclasses.dataclass
class Taxon:
    kingdom: str
    phylum: str
    cls: str
    order: str
    family: str
    genus: str
    species: str

    def __post_init__(self):
        for rank in taxon_ranks:
            value = getattr(self, rank)
            setattr(self, rank, clean_rank(value))

        # Remove names and years
        self.species = clean_name(self.species)

        # Metazoa -> Animalia, Archaeplastida -> Plantae, Chloroplastida -> Plantae
        if self.kingdom == "metazoa":
            self.kingdom = "animalia"
        if self.kingdom == "archaeplastida":
            self.kingdom = "plantae"
        if self.kingdom == "chloroplastida":
            self.kingdom = "plantae"

        # If species contains the genus, fix that.
        if self.species:  # Check that species is not empty.
            maybe_genus, *rest = self.species.split()

            # See if genus is in species label
            if maybe_genus == self.genus and rest:
                # Remove the genus from the species
                self.species = " ".join(rest)

            # See if genus is in species label and genus is missing
            if not self.genus and maybe_genus and rest:
                # Remove the genus from the species
                self.species = " ".join(rest)
                # Add genus
                self.genus = maybe_genus

            # If species is duplicated, fix that
            if len(self.species.split()) == 2:
                first, second = self.species.split()
                if first == second:
                    self.species = first

        for rank in taxon_ranks:
            value = getattr(self, rank)
            setattr(self, rank, clean_rank(value))

    @property
    def empty(self):
        return not (
            self.kingdom
            or self.phylum
            or self.cls
            or self.order
            or self.family
            or self.genus
            or self.species
        )

    @property
    def filled(self):
        return (
            self.kingdom
            and self.phylum
            and self.cls
            and self.order
            and self.family
            and self.genus
            and self.species
        )

    @property
    def taxonomic(self):
        name = " ".join(
            [
                self.kingdom.capitalize(),
                self.phylum.capitalize(),
                self.cls.capitalize(),
                self.order.capitalize(),
                self.family.capitalize(),
                self.genus.capitalize(),
                self.species.lower(),
            ]
        ).strip()

        # Remove double whitespaces
        return " ".join(name.split())

    @property
    def scientific(self):
        name = " ".join(
            [
                self.genus.capitalize(),
                self.species.lower(),
            ]
        ).strip()

        # Remove double whitespaces and make sure the first letter is capitalized.
        return " ".join(name.split()).capitalize()

    @property
    def tagged(self):
        return [
            (key, value)
            for key, value in [
                ("kingdom", self.kingdom.capitalize()),
                ("phylum", self.phylum.capitalize()),
                ("class", self.cls.capitalize()),
                ("order", self.order.capitalize()),
                ("family", self.family.capitalize()),
                ("genus", self.genus.capitalize()),
                ("species", self.species.lower()),
            ]
            if value
        ]

    def to_tuple(self):
        return (
            self.kingdom.capitalize(),
            self.phylum.capitalize(),
            self.cls.capitalize(),
            self.order.capitalize(),
            self.family.capitalize(),
            self.genus.capitalize(),
            self.species.lower(),
        )

    def to_dict(self):
        return {
            "kingdom": self.kingdom.capitalize(),
            "phylum": self.phylum.capitalize(),
            "cls": self.cls.capitalize(),
            "order": self.order.capitalize(),
            "family": self.family.capitalize(),
            "genus": self.genus.capitalize(),
            "species": self.species.lower(),
        }

    def better(self, other):
        """
        Returns if self is a strictly more informative taxon than other.
        """
        self_tags = dict(self.tagged)
        other_tags = dict(other.tagged)

        # Look for tags present in other that are different in self
        for tag, self_value in self_tags.items():
            other_value = other_tags.get(tag, None)
            if other_value and other_value != self_value:
                # Other name has a non-null tag that is different.
                # Can't decide which is better, self or other
                return False

        # Look for tags missing in other that are present in self
        for tag, other_value in other_tags.items():
            self_value = self_tags.get(tag, None)
            if not other_value and self_value:
                # We have a value that other is missing
                return True

        return False


class NameLookup:
    """Lookup from a key to taxonomic name.
    Meant to be subclassed for each different data source.
    """

    def taxon(self, key: object):  # Taxon | None
        """Returns a Taxon object"""
        raise NotImplementedError()

    def keys(self):  # list[object]
        raise NotImplementedError()


class BioscanNameLookup(NameLookup):
    def __init__(self):
        self.lookup = {}  # dict[str, Taxon]

        with open(disk.bioscan_metadata_jsonld) as fd:
            bioscan_metadata = json.load(fd)

        for row in tqdm(bioscan_metadata, desc="Loading Bioscan metadata"):
            taxon = Taxon(
                "Animalia",
                "Arthropoda",
                "Insecta",
                row["order"] if row["order"] != "not_classified" else "",
                row["family"] if row["family"] != "not_classified" else "",
                row["genus"] if row["genus"] != "not_classified" else "",
                row["species"] if row["species"] != "not_classified" else "",
            )
            self.lookup[row["image_file"]] = taxon

    def taxon(self, key):
        return self.lookup.get(key)

    def keys(self):
        return list(self.lookup.keys())


class iNat21NameLookup(NameLookup):
    def __init__(self, inat21_root="/fs/ess/PAS2136/foundation_model/inat21/raw/train"):
        self._taxa = {}

        for clsdir in os.listdir(inat21_root):
            index, king, phyl, cls, ord, fam, genus, species = clsdir.split("_")
            index = int(index, base=10)
            taxon = Taxon(king, phyl, cls, ord, fam, genus, species)
            self._taxa[index] = taxon

    def taxon(self, key):
        return self._taxa.get(key)

    def keys(self):  # list[int]
        return list(self._taxa.keys())


class CommonNameLookup(dict):
    def __init__(
        self,
    ):
        # ITIS
        for row in helpers.csvreader(disk.itis_hierarchy_csv):
            # Create a Taxon object because it normalizes and cleans
            taxon = Taxon(
                row["Kingdom"],
                row["Phylum"],
                row["Class"],
                row["Order"],
                row["Family"],
                row["Genus"],
                row["Species"],
            )

            # These are not always the same length.
            commons = row["terminal_vernacular"].split(", ")
            langs = row["terminal_vernacular_lang"].split(", ")

            for com, lang in zip(commons, langs):
                if com and lang == "English" and id not in self:
                    self[taxon.scientific] = com
                    break

        # iNaturalist
        inaturalist_id_to_taxon = {}
        inaturalist_id_to_scientific = {}

        for row in helpers.csvreader(disk.inaturalist_taxa_csv):
            # Create a Taxon object because it normalizes and cleans
            taxon = Taxon(
                row["kingdom"],
                row["phylum"],
                row["class"],
                row["order"],
                row["family"],
                row["genus"],
                row["specificEpithet"],
            )
            if row["taxonRank"] in ("species", "variety"):
                inaturalist_id_to_taxon[int(row["id"])] = taxon
                inaturalist_id_to_scientific[int(row["id"])] = row["scientificName"]

        for row in helpers.csvreader(disk.inaturalist_vernacularnames_csv):
            name = row["vernacularName"]
            if not name:
                continue

            taxon = inaturalist_id_to_taxon.get(int(row["id"]))
            if taxon and taxon.scientific not in self:
                self[taxon.scientific] = name

            scientific = inaturalist_id_to_scientific.get(int(row["id"]))
            if scientific and scientific not in self:
                self[scientific] = name

        # EOL
        # First pass, only use preferred names
        for row in helpers.csvreader(disk.eol_vernacularnames_csv):
            if row["language_code"] != "eng":  # only keep english
                continue

            if not bool(row["is_preferred_by_eol"]):
                continue

            common = clean_name(row["vernacular_string"])
            scientific = strip_html(row["canonical_form"])

            if scientific and common and scientific not in self:
                self[scientific] = common

        # Second pass, fill in any missing values
        for row in helpers.csvreader(disk.eol_vernacularnames_csv):
            if row["language_code"] != "eng":  # only keep english
                continue

            common = clean_name(row["vernacular_string"])
            scientific = strip_html(row["canonical_form"])

            if scientific and common and scientific not in self:
                self[scientific] = common


class NameUpgrader:
    """ """

    known_duplicate_values = {
        "actinobacteria",
        "aquificae",
        "chlamydiae",
        "chrysiogenetes",
        "collembola",
        "deferribacteres",
        "diplura",
        "elusimicrobia",
        "gemmatimonadetes",
        "micrognathozoa",
        "nitrospira",
        "protura",
        "protozoa",
        "thermodesulfobacteria",
        "thermotogae",
        ("leptogaster", "flavipes"),
    }

    def __init__(self):
        # List of all changes made during upgrade()
        self.changes = collections.Counter()

        # Known exceptions
        self._lookup = {
            "collembola": (("cls", "order"), "arthropoda"),
            "diplura": (("cls", "order"), "arthropoda"),
            "protura": (("cls", "order"), "arthropoda"),
            "actinobacteria": (("phylum", "cls"), "bacteria"),
            "aquificae": (("phylum", "cls"), "bacteria"),
            "chlamydiae": (("phylum", "cls"), "bacteria"),
            "chrysiogenetes": (("phylum", "cls"), "bacteria"),
            "deferribacteres": (("phylum", "cls"), "bacteria"),
            "elusimicrobia": (("phylum", "cls"), "bacteria"),
            "gemmatimonadetes": (("phylum", "cls"), "bacteria"),
            "micrognathozoa": (("phylum", "cls"), "animalia"),
            "nitrospira": (("phylum", "cls"), "bacteria"),
            "protozoa": (("kingdom", "phylum"), None),
            "thermodesulfobacteria": (("phylum", "cls"), "bacteria"),
            "thermotogae": (("phylum", "cls"), "bacteria"),
            ("leptogaster", "flavipes"): ("scientific", "asilidae"),
        }

        assert (
            set(self._lookup.keys()) == self.known_duplicate_values
        ), "Update either self._lookup or self.known_duplicate_values"

        self._errors = collections.Counter()

        for row in helpers.csvreader(disk.itis_hierarchy_csv):
            taxon = Taxon(
                row["Kingdom"],
                row["Phylum"],
                row["Class"],
                row["Order"],
                row["Family"],
                row["Genus"],
                row["Species"],
            )
            self._add_taxon(taxon)

        if self._errors:
            print(self._errors)

        # Use resolved.jsonl
        with open(disk.resolved_jsonl) as fd:
            for line in fd:
                line = json.loads(line)
                scientific = next(iter(line.keys()))
                genus, *rest = scientific.lower().split()
                species = " ".join(rest)

                ranks = next(iter(line[scientific].values()))

                taxon = Taxon(
                    ranks.get("kingdom", ""),
                    ranks.get("phylum", ""),
                    ranks.get("class", ""),
                    ranks.get("order", ""),
                    ranks.get("family", ""),
                    genus,
                    species,
                )

                self._add_taxon(taxon, handle_existing="old")

        if self._errors:
            breakpoint()

    def _add_taxon(self, taxon, handle_existing="raise"):
        """
        Adds a taxon to the lookup.
        taxon: Taxon object
        handle_existing: either 'raise', 'old', 'new', or 'log'. In the case of an existing key that is differet:
        * 'raise' will raise an error.
        * 'old' will ignore the new key and use the old key.
        * 'new' will ignore the old key and use the new key.
        * 'log' will ignore the new key and store the error in self._errors
        """

        # Known errors
        if taxon.family == "charinidae" and taxon.order == "squamata":
            taxon.family = "boidae"

        for rank, value, parent in [
            ("scientific", (taxon.genus, taxon.species), taxon.family),
            ("family", taxon.family, taxon.order),
            ("order", taxon.order, taxon.cls),
            ("cls", taxon.cls, taxon.phylum),
            ("phylum", taxon.phylum, taxon.kingdom),
            ("kingdom", taxon.kingdom, None),
        ]:
            if not value:
                continue

            # Known typo
            if value == "arthropda":
                value == "arthropoda"

            if value in self._lookup:
                # Matches, so no problem.
                if self._lookup[value] == (rank, parent):
                    continue

                # We know about this error already.
                if value in self.known_duplicate_values:
                    continue

                if handle_existing == "raise":
                    raise ValueError(
                        f"{self._lookup[value]} != {(rank, parent)} for {value}"
                    )
                elif handle_existing == "old":
                    continue
                elif handle_existing == "new":
                    pass
                elif handle_existing == "log":
                    self._errors[(value, self._lookup[value], (rank, parent))] += 1
                    continue
                else:
                    raise ValueError(handle_existing)

            self._lookup[value] = (rank, parent)

    def fill(self, ranks, name):
        """
        Fills ranks with
        ranks (dict[str, str]): initial filled in ranks. Maps from taxon_ranks to values.
        name (str): initial name to start filling in from.

        Returns ranks, which is also modified by this method.
        """

        def set_rank(key, value):
            """Fills in a key-value pair in ranks. If ranks[key] != value, make a note of it in self.changes."""
            existing = ranks.get(key, "")
            if existing and existing != value:
                self.changes[(key, existing, value)] += 1

            ranks[key] = value

        # Recursively go up the chain to fill the name in
        while name in self._lookup:
            rank, parent_name = self._lookup[name]
            if isinstance(rank, tuple):
                for r in rank:
                    set_rank(r, name)
            elif rank in taxon_ranks:
                set_rank(rank, name)
            name = parent_name

        return ranks

    def upgrade(self, taxon):
        name, rank = find_initial_name(taxon)

        while True:
            ranks = {rank: getattr(taxon, rank) for rank in taxon_ranks}
            ranks = self.fill(ranks, name)
            taxon = Taxon(**ranks)
            if taxon.filled:
                return taxon

            name, rank = find_initial_name(taxon, failed=rank)
            if (name, rank) == (None, None):
                return Taxon(**ranks)

        return Taxon(**ranks)


def find_initial_name(taxon: Taxon, failed=None) -> tuple[str, str]:
    """
    Problem: kingdom and phylum is filled in, cls is missing.
    upgrade sees that kingdom is not empty and uses that as top rank.
    Really we want the highest rank that is continuous to species
    But then what happens with K, P, _, O, F, _, _? -> want to use O
    K, P, _, O, F, G, S -> use O
    K, P, _, O, F, _, S -> use O, can't fill in genus
    K, P, _, _, F, _, S -> use F, can't fill in genus
    K, _, C, _, F, G, S -> use F

    TODO: use the species and find all possible genuses, then use those to compare against families or higher ranks
    """
    if failed == "scientific":
        # Can't give anything further down the tree
        return None, None

    # going from scientific up to kingdom
    current = ((taxon.genus, taxon.species), "scientific")
    for rank in reversed(taxon_ranks[:-2]):
        if rank == failed:
            break
        if getattr(taxon, rank):
            current = (getattr(taxon, rank), rank)
        else:
            return current

    return current


def strip_html(string):
    """Just removes <i>...</i> tags and similar. Not complete."""

    def clean(s):
        return re.sub(r"<.*?>(.*?)</.*?>", r"\1", s)

    while string != clean(string):
        string = clean(string)

    return string


def clean_name(name):
    name = strip_html(name.lower()).strip()

    patterns = [
        r"\(",  # Remove anything after parens
        r"ssp\.",  # ssp. indicates subspecies
        r"subsp\.",  # subsp. indicates subspecies
        r"var\.",  # var. indicates variant
        r"\w+, \w+ & \w+",  # name, name & name
        r"\w+ & \w+",  # name & name
        r"\w+ \d\d\d\d",  # name year
        r"\w+, \d\d\d\d",  # name, year
        r"\d\d\d\d",  # year
        r" [A-z]\.",  # initial
        r"<br ?>",  # linebreak
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            start, end = match.span()
            name = name[:start].strip()

    return name


def load_name_lookup(path, keytype=str):
    """
    Returns dict[key, (taxonomic, common)]
    """
    with open(path) as fd:
        return {
            keytype(key): (Taxon(*taxa), common, classes)
            for key, (taxa, common, classes) in json.load(fd).items()
        }


def get_common(taxon, common):
    if common:
        return common
    if taxon.scientific:
        return taxon.scientific

    return taxon.taxonomic
