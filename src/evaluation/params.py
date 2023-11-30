import argparse
import ast
import os

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split("=")
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                # fallback to string (avoid need to escape on command line)
                kw[key] = str(value)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_filename",
        type=str,
        default=None,
        help="File path of the CSV annotation file under the base folder in --data_root.",
    )
    parser.add_argument(
        "--text_type",
        type=str,
        default='asis',
        help="Text type of annotations for test examples.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["retrieve", "eval", "all"],
        default='all',
        help="retrieve: retrieve image embeddings from image encoder; eval: evaluate with pickle file containing image embeddings; all: evaluate from scratch which both retrieve image embeddings and evaluate the embeddings in one run.",
    )
    parser.add_argument(
        "--nfold",
        type=int,
        default=5,
        help="The number of times of sampling training examples during few-shot.",
    )
    parser.add_argument(
        "--kshot_list",
        type=int,
        nargs="+",
        default=[1,5],
        help="A list of integers for k in k-shot.",
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        default='', 
        help="File path of base folder which contains images and a CSV annotation file. "
    )
    parser.add_argument(
        "--logs",
        type=str,
        default=None,
        help="Where to store logs. Use None or 'none' to avoid storing logs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of dataloader workers per GPU."
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
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-16",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default="openai",
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path. If running few-shot.py with --task_type=eval, use this parameter as the pickle file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action="store_true",
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--image-mean",
        type=float,
        nargs="+",
        default=None,
        metavar="MEAN",
        help="Override default image mean value of dataset",
    )
    parser.add_argument(
        "--image-std",
        type=float,
        nargs="+",
        default=None,
        metavar="STD",
        help="Override default image std deviation of of dataset",
    )
    parser.add_argument("--aug-cfg", nargs="*", default={}, action=ParseKwargs)
    parser.add_argument(
        "--force-image-size",
        type=int,
        nargs="+",
        default=None,
        help="Override default image size",
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action="store_true",
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action="store_true",
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action="store_true",
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action="store_true",
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0, 
        help="Default random seed."
    )
    args, _ = parser.parse_known_args(args)

    return args
