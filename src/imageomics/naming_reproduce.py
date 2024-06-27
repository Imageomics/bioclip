import dataclasses
import json
import logging
import re


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
