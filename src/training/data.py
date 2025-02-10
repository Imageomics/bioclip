import logging
import random
from dataclasses import dataclass

from PIL import Image
import torch
from torch.utils.data import (
    DataLoader,
    IterableDataset,
)
from torch.utils.data import Sampler
from open_clip.tokenizer import HFTokenizer, SimpleTokenizer

from typing import List, Optional, Callable, Iterator, Tuple, Protocol, cast
import webdataset as wds
import io
import polars as pl
from datasets import load_dataset, DatasetDict
import requests
from pydantic import BaseModel, ValidationError
from enum import Enum
from tqdm import tqdm


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: Optional[Sampler] = None


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TaxonomicFilter(Protocol):
    def __call__(self, taxonomic_name: "TaxonomicName") -> bool: ...


class Rank(Enum):
    KINGDOM = 0
    PHYLUM = 1
    CLASS = 2
    ORDER = 3
    FAMILY = 4
    GENUS = 5
    SPECIES = 6

    def get_label(self):
        return self.name.lower()


class TaxonomicName(BaseModel):
    kingdom: str
    phylum: Optional[str]
    class_: Optional[str]
    order: Optional[str]
    family: Optional[str]
    genus: Optional[str]
    species: Optional[str]
    common_names: List[str] = []

    def fix(self):
        if not self.species or " " not in self.species:
            return
        self.species = self.species.split(" ")[1]

    @classmethod
    def from_list(cls, v: List[str]) -> "TaxonomicName":
        return cls(
            kingdom=v[0],
            phylum=v[1],
            class_=v[2],
            order=v[3],
            family=v[4],
            genus=v[5],
            species=v[6],
        )

    def rank(self, rank: Rank) -> str:
        taxonomy_list = [
            self.kingdom,
            self.phylum,
            self.class_,
            self.order,
            self.family,
            self.genus,
            self.species,
        ]
        return taxonomy_list[rank.value] if taxonomy_list[rank.value] else ""

    @property
    def scientific_name(self) -> str:
        self.fix()
        return f"{self.genus or ''} {self.species or ''}".strip()

    @property
    def taxonomic_name(self) -> str:
        taxonomy_list = [
            self.kingdom,
            self.phylum,
            self.class_,
            self.order,
            self.family,
            self.genus,
            self.species,
        ]
        return " ".join([str(t) for t in taxonomy_list if t])

    @property
    def common_name(self) -> str:
        if self.common_names:
            return random.choice(self.common_names)
        return ""

    def __str__(self) -> str:
        self.fix()
        string = " ".join([self.rank(r) for r in Rank])
        return string

    def __hash__(self) -> int:
        return hash(self.scientific_name)

    def __eq__(self, other) -> bool:
        return self.scientific_name == other.scientific_name

    @staticmethod
    def from_dict(input: dict) -> "TaxonomicName | None":
        common_keys = ("common_name", "common")
        common_names: List[str] = [
            input[k] for k in common_keys if k in input and input[k] is not None
        ]
        kingdom = input.get("kingdom")
        assert kingdom is not None, "Kingdom is required"
        try:
            return TaxonomicName(
                kingdom=kingdom,
                phylum=input.get("phylum"),
                class_=input.get("class"),
                order=input.get("order"),
                family=input.get("family"),
                genus=input.get("genus"),
                species=input.get("species"),
                common_names=common_names,
            )
        except ValidationError as e:
            logging.error(
                f"Error creating TaxonomicName from dict: {input}. Error: {e}"
            )
            return None


def create_tokens(
    taxonomic_name: TaxonomicName, tokenizer: HFTokenizer | SimpleTokenizer
) -> torch.Tensor:
    text_types = ["scientific_name", "common_name", "taxonomic_name"]
    text_type = random.choice(text_types)
    if text_type == "scientific_name":
        text = taxonomic_name.scientific_name
    elif text_type == "common_name":
        text = taxonomic_name.common_name
    elif text_type == "taxonomic_name":
        text = taxonomic_name.taxonomic_name
    else:
        raise ValueError(f"Unknown text type: {text_type}")

    if not text:
        text = taxonomic_name.scientific_name

    return tokenizer(f"a photo of a {text}")


class TaxonomyFilter:
    def __init__(self, rank: Rank, value: str):
        self.rank = rank
        self.value = value

    def __call__(self, taxonomic_name: "TaxonomicName") -> bool:
        return taxonomic_name.rank(self.rank) == self.value


class TreeOfLifeDataset(IterableDataset):
    default_urls = [
        f"https://huggingface.co/datasets/imageomics/TreeOfLife-10M/resolve/343c36b9b362494065ac427ac1da989b834b22cf/dataset/EOL/image_set_{i:02d}.tar.gz"
        for i in range(1, 64)
    ]
    default_labels_url = "https://huggingface.co/datasets/imageomics/TreeOfLife-10M/resolve/main/metadata/catalog.csv"

    def __init__(
        self,
        tokenizer: HFTokenizer | SimpleTokenizer,
        urls: List[str] = default_urls,
        labels_url: str = default_labels_url,
        transform: Optional[Callable] = None,
        filters: List[TaxonomicFilter] = [],
        seed: int = 42,
        split: str = "train",
    ):
        self.transform = transform
        self.tokenizer = tokenizer
        random.seed(seed)

        self.labels = pl.read_csv(labels_url)
        self.label_map = dict(
            zip(
                self.labels["treeoflife_id"],
                self.labels.drop("treeoflife_id").to_dicts(),
            )
        )
        self.filters = filters
        self.urls = random.sample(urls, len(urls))
        self.split = split

    def __iter__(self) -> Iterator[Tuple[Image.Image, torch.Tensor]]:
        for url in self.urls:
            try:
                dataset = wds.WebDataset(url, shardshuffle=True)
                for sample in tqdm(dataset, desc=f"Processing {url}", leave=False):
                    label = self.label_map.get(sample["__key__"], None)
                    if label is None:
                        logging.warning(f"Missing label for key: {sample['__key__']}")
                        continue

                    if self.split not in label["split"]:
                        continue

                    taxonomic_name = TaxonomicName.from_dict(label)
                    if taxonomic_name is None:
                        continue

                    filter_pass = True
                    for f in self.filters:
                        if not f(taxonomic_name):
                            filter_pass = False
                            break

                    if not filter_pass:
                        continue

                    try:
                        img = Image.open(io.BytesIO(sample["jpg"]))
                    except Exception as e:
                        logging.error(
                            f"Error opening image for key {sample['__key__']}: {e}"
                        )
                        continue

                    yield (
                        self.transform(img) if self.transform else img,
                        create_tokens(taxonomic_name, self.tokenizer),
                    )
            except Exception as e:
                logging.error(f"Error processing URL {url}: {e}")


class BioTroveDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: HFTokenizer | SimpleTokenizer,
        transform: Optional[Callable] = None,
        filters: List[TaxonomicFilter] = [],
        split: str = "train",
    ):
        self.transform = transform
        self.split = split
        self.ds: DatasetDict = cast(
            DatasetDict, load_dataset("BGLab/BioTrove", streaming=True, split=split)
        )
        self.filters = filters
        self.tokenizer = tokenizer

    def __iter__(self) -> Iterator[Tuple[Image.Image, torch.Tensor]]:
        for row in tqdm(self.ds[self.split], desc="Processing BioTrove", leave=False):
            row = dict(row)
            taxonomic_name = TaxonomicName.from_dict(row)
            if taxonomic_name is None:
                continue

            filter_pass = True
            for f in self.filters:
                if not f(taxonomic_name):
                    filter_pass = False
                    break

            if not filter_pass:
                continue

            try:
                img_bytes = requests.get(
                    row["photo_url"], stream=True, timeout=10
                ).content
                img = Image.open(io.BytesIO(img_bytes))
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching image from {row['photo_url']}: {e}")
                continue
            except Exception as e:
                logging.error(f"Error opening image: {e}")
                continue

            yield (
                self.transform(img) if self.transform else img,
                create_tokens(taxonomic_name, self.tokenizer),
            )


def get_data(
    dataset: str,
    tokenizer: HFTokenizer | SimpleTokenizer,
    transform: Optional[Callable] = None,
    filters: List[TaxonomicFilter] = [],
    seed: int = 42,
    batch_size: int = 32,
    split: str = "train",
) -> DataInfo:
    if dataset == "TreeOfLife":
        return DataInfo(
            DataLoader(
                TreeOfLifeDataset(
                    transform=transform,
                    filters=filters,
                    seed=seed,
                    split=split,
                    tokenizer=tokenizer,
                ),
                batch_size=batch_size,
            )
        )
    elif dataset == "BioTrove":
        return DataInfo(
            DataLoader(
                BioTroveDataset(
                    transform=transform,
                    filters=filters,
                    split=split,
                    tokenizer=tokenizer,
                ),
                batch_size=batch_size,
            )
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
