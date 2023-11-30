import collections
import dataclasses
import json
import logging
import os
import re

from tqdm import tqdm

from . import disk, helpers

logger = logging.getLogger()

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def dataset_class_to_taxon(cls):
    tiers = cls.split("_")
    if cls[0].isdigit():
        index, *tiers = cls.split("_")
        index = int(index, base=10)
        return Taxon(*tiers, dataset_id=index)
    else:
        return Taxon(*cls.split("_"))

@dataclasses.dataclass
class Taxon:
    kingdom: str = ""
    phylum: str = ""
    cls: str = ""
    order: str = ""
    family: str = ""
    genus: str = ""
    species: str = ""
    subspecies: str = ""
    # id from inaturalist.org (taxa.csv)
    website_id: int = -1
    # id from the inat21 dataset
    dataset_id: int = -1
    # common name (from VernacularNames-english.csv)
    common_name: str = ""

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
    
    @property
    def scientific_name(self):
        if not self.subspecies == '':
            return self.genus.capitalize() + ' ' + self.species.lower() + ' ' + self.subspecies.lower()
        elif not self.species == '':
            return self.genus.capitalize() + ' ' + self.species.lower()
        elif not self.genus == '':
            return self.genus.capitalize()
        else:
            return None
    
    @property
    def taxonomic_name(self):
        if not self.subspecies == '':
            if self.kingdom == "":
                return " ".join([
                    self.genus.capitalize(),
                    self.species.lower(),
                    self.subspecies.lower()
                ])
            else:
                return " ".join([
                    self.kingdom.capitalize(),
                    self.phylum.capitalize(),
                    self.cls.capitalize(),
                    self.order.capitalize(),
                    self.family.capitalize(),
                    self.genus.capitalize(),
                    self.species.lower(),
                    self.subspecies.lower()
                ])
        else:
            if self.kingdom == "":
                return " ".join([
                    self.genus.capitalize(),
                    self.species.lower(),
                ])
            else:
                return " ".join([
                    self.kingdom.capitalize(),
                    self.phylum.capitalize(),
                    self.cls.capitalize(),
                    self.order.capitalize(),
                    self.family.capitalize(),
                    self.genus.capitalize(),
                    self.species.lower(),
                ])
    @property
    def get_common_name(self):
        return self.common_name

    @property
    def sci_common_name(self):
        if self.common_name == '':
            return self.scientific_name
        else:
            return self.scientific_name + ' with common name ' + self.get_common_name
    
    @property
    def taxon_common_name(self):
        if self.common_name == '':
            return self.taxonomic_name
        else:
            return self.taxonomic_name + ' with common name ' + self.get_common_name

def to_classes(data,text_type):
    if text_type == 'asis':
        return data['class']

    data_view = data.drop(columns=list(set(data.keys()) - set(Taxon.__dict__.keys())))

    if text_type == 'sci':
        return data_view.apply(lambda x: Taxon(**x.to_dict()).scientific_name, axis=1).values.tolist()
    elif text_type == 'taxon':
        return data_view.apply(lambda x: Taxon(**x.to_dict()).taxonomic_name, axis=1).values.tolist()
    elif text_type == 'com':
        return data_view.apply(lambda x: Taxon(**x.to_dict()).get_common_name, axis=1).values.tolist()
    elif text_type == 'sci_com':
        return data_view.apply(lambda x: Taxon(**x.to_dict()).sci_common_name, axis=1)
    elif text_type == 'taxon_com':
        return data_view.apply(lambda x: Taxon(**x.to_dict()).taxon_common_name, axis=1)
    else:
        raise ValueError("text type is not acceptable.")
