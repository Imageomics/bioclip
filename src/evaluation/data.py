import os
import random
import json
import pandas as pd
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from ..imageomics import naming_eval


def make_splits(directory) -> dict[str, list[str]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    random.seed(1337)
    shuffled = random.sample(classes, k=len(classes))

    # Check that they're the same order on all machines
    # err = "not reproducible"
    # assert (
    #     shuffled[0]
    #     == "08737_Plantae_Tracheophyta_Magnoliopsida_Lamiales_Verbenaceae_Verbena_hastata"
    # ), err
    # assert (
    #     shuffled[8000]
    #     == "03011_Animalia_Chordata_Amphibia_Anura_Microhylidae_Kaloula_pulchra"
    # ), err
    # assert (
    #     shuffled[8999]
    #     == "02422_Animalia_Arthropoda_Insecta_Odonata_Gomphidae_Arigomphus_furcifer"
    # ), err
    # assert (
    #     shuffled[9000]
    #     == "01963_Animalia_Arthropoda_Insecta_Lepidoptera_Nymphalidae_Pyronia_bathseba"
    # ), err
    # assert (
    #     shuffled[9999]
    #     == "06333_Plantae_Tracheophyta_Liliopsida_Poales_Poaceae_Avena_fatua"
    # ), err

    return {
        "pretraining": shuffled[:],
        "seen": shuffled[:],
        "unseen": shuffled[:],
    }


class PretrainingData(datasets.ImageFolder):
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        """
        Only chooses classes that are unseen during pretraining
        """
        splits = make_splits(directory)
        classes = splits["pretraining"]

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class UnseenData(datasets.ImageFolder):
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        """
        Only chooses classes that are unseen during pretraining
        """
        splits = make_splits(directory)
        classes = splits["unseen"]

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class SeenData(datasets.ImageFolder):
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        """
        Only chooses classes that are seen during pretraining
        """
        splits = make_splits(directory)
        classes = splits["seen"]

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

def img_loader(filepath):
    img = Image.open(filepath)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    return img

class DatasetFromFile(Dataset):
    def __init__(self, filepath, label_filepath=None, transform=None, classes='asis'):
        super(DatasetFromFile, self).__init__()
        self.basefilepath = filepath
        if label_filepath is None:
            label_filepath = os.path.join(self.basefilepath, 'metadata.csv')
        else:
            label_filepath = os.path.join(self.basefilepath, label_filepath)

        self.data = pd.read_csv(label_filepath, index_col=0).fillna('')
        self.transform = transform
        self.data['class'] = naming_eval.to_classes(self.data,classes)
        self.classes = self.data['class'].unique()
        # create class_to_idx dict
        if 'class_idx' in self.data.columns:
            self.class_to_idx = dict([(x, y) for x, y in zip(self.data['class'], self.data['class_idx'])])
        else:
            self.class_to_idx = dict(zip(self.classes,range(len(self.classes))))
            self.data['class_idx'] = self.data['class'].apply(lambda x: self.class_to_idx[x])

        self.idx_to_class = dict([(v,k)for k,v in self.class_to_idx.items()])
        self.samples = self.data['filepath'].values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        filepath = os.path.join(self.basefilepath,item['filepath'])
        img = img_loader(filepath)
        if self.transform is not None:
            img = self.transform(img)
        
        target = item['class_idx']
        return img, target
