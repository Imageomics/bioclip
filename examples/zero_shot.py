"""
Do zero-shot image classification.

Writes the output to a plaintext and JSON format in the logs directory.
"""
import argparse
import ast
import contextlib
import json
import logging
import os
import random
import sys

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torchvision import datasets
from tqdm import tqdm

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("main")

openai_templates = [
    lambda c: f"a bad photo of a {c}.",
    lambda c: f"a photo of many {c}.",
    lambda c: f"a sculpture of a {c}.",
    lambda c: f"a photo of the hard to see {c}.",
    lambda c: f"a low resolution photo of the {c}.",
    lambda c: f"a rendering of a {c}.",
    lambda c: f"graffiti of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a cropped photo of the {c}.",
    lambda c: f"a tattoo of a {c}.",
    lambda c: f"the embroidered {c}.",
    lambda c: f"a photo of a hard to see {c}.",
    lambda c: f"a bright photo of a {c}.",
    lambda c: f"a photo of a clean {c}.",
    lambda c: f"a photo of a dirty {c}.",
    lambda c: f"a dark photo of the {c}.",
    lambda c: f"a drawing of a {c}.",
    lambda c: f"a photo of my {c}.",
    lambda c: f"the plastic {c}.",
    lambda c: f"a photo of the cool {c}.",
    lambda c: f"a close-up photo of a {c}.",
    lambda c: f"a black and white photo of the {c}.",
    lambda c: f"a painting of the {c}.",
    lambda c: f"a painting of a {c}.",
    lambda c: f"a pixelated photo of the {c}.",
    lambda c: f"a sculpture of the {c}.",
    lambda c: f"a bright photo of the {c}.",
    lambda c: f"a cropped photo of a {c}.",
    lambda c: f"a plastic {c}.",
    lambda c: f"a photo of the dirty {c}.",
    lambda c: f"a jpeg corrupted photo of a {c}.",
    lambda c: f"a blurry photo of the {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a good photo of the {c}.",
    lambda c: f"a rendering of the {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"a photo of one {c}.",
    lambda c: f"a doodle of a {c}.",
    lambda c: f"a close-up photo of the {c}.",
    lambda c: f"a photo of a {c}.",
    lambda c: f"the origami {c}.",
    lambda c: f"the {c} in a video game.",
    lambda c: f"a sketch of a {c}.",
    lambda c: f"a doodle of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a low resolution photo of a {c}.",
    lambda c: f"the toy {c}.",
    lambda c: f"a rendition of the {c}.",
    lambda c: f"a photo of the clean {c}.",
    lambda c: f"a photo of a large {c}.",
    lambda c: f"a rendition of a {c}.",
    lambda c: f"a photo of a nice {c}.",
    lambda c: f"a photo of a weird {c}.",
    lambda c: f"a blurry photo of a {c}.",
    lambda c: f"a cartoon {c}.",
    lambda c: f"art of a {c}.",
    lambda c: f"a sketch of the {c}.",
    lambda c: f"a embroidered {c}.",
    lambda c: f"a pixelated photo of a {c}.",
    lambda c: f"itap of the {c}.",
    lambda c: f"a jpeg corrupted photo of the {c}.",
    lambda c: f"a good photo of a {c}.",
    lambda c: f"a plushie {c}.",
    lambda c: f"a photo of the nice {c}.",
    lambda c: f"a photo of the small {c}.",
    lambda c: f"a photo of the weird {c}.",
    lambda c: f"the cartoon {c}.",
    lambda c: f"art of the {c}.",
    lambda c: f"a drawing of the {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a black and white photo of a {c}.",
    lambda c: f"the plushie {c}.",
    lambda c: f"a dark photo of a {c}.",
    lambda c: f"itap of a {c}.",
    lambda c: f"graffiti of the {c}.",
    lambda c: f"a toy {c}.",
    lambda c: f"itap of my {c}.",
    lambda c: f"a photo of a cool {c}.",
    lambda c: f"a photo of a small {c}.",
    lambda c: f"a tattoo of the {c}.",
]


def parse_args(args):
    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            kw = {}
            for value in values:
                key, value = value.split("=")
                try:
                    kw[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    # fallback to string (avoid need to escape on command line)
                    kw[key] = str(value)
            setattr(namespace, self.dest, kw)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        nargs="+",
        help="Path to dirs(s) with validation data. In the format NAME=PATH.",
        action=ParseKwargs,
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="Where to write logs"
    )
    parser.add_argument(
        "--exp", type=str, default="bioclip-zero-shot", help="Experiment name."
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Default random seed.")
    args = parser.parse_args(args)
    os.makedirs(os.path.join(args.logs, args.exp), exist_ok=True)

    return args


def make_txt_features(model, classnames, templates, args):
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    with torch.no_grad():
        txt_features = []
        for classname in tqdm(classnames):
            classname = " ".join(word for word in classname.split("_") if word)
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(args.device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            txt_features.append(class_embedding)
        txt_features = torch.stack(txt_features, dim=1).to(args.device)
    return txt_features


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0, keepdim=True).item() for k in topk]


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return contextlib.suppress


def run(model, txt_features, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = open_clip.get_cast_dtype(args.precision)

    top1, top5, n = 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            targets = targets.to(args.device)

            with autocast():
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = model.logit_scale.exp() * image_features @ txt_features

            # Measure accuracy
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = top1 / n
    top5 = top5 / n
    return top1, top5


def evaluate(model, data, args):
    results = {}

    logger.info("Starting zero-shot classification.")

    for split in data:
        logger.info("Building zero-shot %s classifier.", split)

        classnames = data[split].dataset.classes
        classnames = [name.replace("_", " ") for name in classnames]

        txt_features = make_txt_features(model, classnames, openai_templates, args)

        logger.info("Got text features.")
        top1, top5 = run(model, txt_features, data[split], args)

        logger.info("%s-top1: %.3f", split, top1 * 100)
        logger.info("%s-top5: %.3f", split, top5 * 100)

        results[f"{split}-top1"] = top1 * 100
        results[f"{split}-top5"] = top5 * 100

        logger.info("Finished zero-shot %s.", split)

    logger.info("Finished zero-shot classification.")

    return results


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Init torch device
    if torch.cuda.is_available():
        device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    args.device = device

    # Random seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load model.
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip"
    )

    # Write datasets
    params_file = os.path.join(args.logs, args.exp, "params.json")
    with open(params_file, "w") as fd:
        params = {name: getattr(args, name) for name in vars(args)}
        json.dump(params, fd, sort_keys=True, indent=4)

    # Initialize datasets.
    data = {}
    for split, path in args.datasets.items():
        data[split] = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transform=preprocess_val),
            batch_size=args.batch_size,
            num_workers=args.workers,
            sampler=None,
            shuffle=False,
        )

    model.eval()
    results = evaluate(model, data, args)

    results_file = os.path.join(args.logs, args.exp, "results.json")
    with open(results_file, "w") as fd:
        json.dump(results, fd, indent=4, sort_keys=True)
