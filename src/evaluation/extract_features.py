"""
Extract features fron iNat21 dataset for visualization.
"""
import datetime
import logging
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from imageomics.naming import iNat21NameLookup
from open_clip import create_model_and_transforms, trace_model
from training.logger import setup_logging

from .params import parse_args
from .utils import init_device, random_seed, save_json


class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        return (img, label, path)


def extract_image_features(model, dloader, device, class_embeddings=None):
    feats = None
    lbls = None
    all_paths = []
    for img, lbl, paths in tqdm(dloader, desc="Extracting Features"):
        img = img.to(device)
        lbl = lbl.cpu().detach().numpy()
        f = model.encode_image(img)
        f = F.normalize(f, dim=-1)
        if class_embeddings is not None:
            f = 100.0 * f @ class_embeddings
        f = f.cpu().detach().numpy()
        if feats is None:
            feats = f
            lbls = lbl
        else:
            feats = np.concatenate((feats, f), axis=0)
            lbls = np.concatenate((lbls, lbl), axis=0)
        all_paths = all_paths + list(paths)

    return feats, lbls, all_paths


def _load_model(args):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem/uri use
        model_name_safe = args.model.replace("/", "-")
        date_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = "-".join(
            [
                date_str,
                f"model_{model_name_safe}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
                "zero_shot_iid",
            ]
        )

    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    os.makedirs(log_base_path, exist_ok=True)
    log_filename = f"out-{args.rank}" if args.log_local else "out.log"
    args.log_path = os.path.join(log_base_path, log_filename)

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup checkpoint logging
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    os.makedirs(args.checkpoint_path, exist_ok=True)

    if (
        isinstance(args.force_image_size, (tuple, list))
        and len(args.force_image_size) == 1
    ):
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]

    random_seed(args.seed, 0)
    model, _, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,  # This is the pretrained weights
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=None,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
    )

    return model, device, preprocess_val


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    parser = ArgumentParser()
    parser.add_argument("--exp_type", type=str, default="openai")
    parser.add_argument(
        "--val_root", type=str, default="/local/scratch/cv_datasets/inat21/raw/val"
    )
    parser.add_argument(
        "--feature_output",
        type=str,
        default="/local/scratch/carlyn.1/clip_paper_bio/features",
    )
    # Be sure to set the --pretrained [path to weights]

    args, _ = parser.parse_known_args(namespace=args)

    model, device, preprocess_val = _load_model(args)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    data_save_path = os.path.join(args.feature_output, args.exp_type + "_features.npy")
    lbl_save_path = os.path.join(args.feature_output, args.exp_type + "_labels.npy")
    path_save_path = os.path.join(args.feature_output, args.exp_type + "_paths.json")

    model.eval()

    # initialize datasets
    dset = ImageFolderWithPaths(args.val_root, transform=preprocess_val)
    classes = dset.classes
    inat_name_lookup = iNat21NameLookup(inat21_root=args.val_root)
    classnames = [inat_name_lookup.taxonomic(cls) for cls in classes]
    dloader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size,
        num_workers=8,
        sampler=None,
    )

    # logging
    args.save_logs = args.logs and args.logs.lower() != "none"

    with torch.no_grad():
        features, lbls, all_paths = extract_image_features(
            model, dloader, device, class_embeddings=None
        )

    np.save(data_save_path, features)
    np.save(lbl_save_path, lbls)
    save_json(path_save_path, all_paths)
