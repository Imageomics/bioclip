"""
Do few-shot classification.

Single-process. If you want to run all evaluations of a single model at once, look in scripts/.
"""

import datetime
import logging
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import json
import pickle
import numpy as np
import random
import shutil
from numpy import linalg as LA
from scipy.stats import mode

from .data import DatasetFromFile
from .params import parse_args
from .utils import init_device, random_seed

from ..open_clip import (
    create_model_and_transforms,
    get_cast_dtype,
    get_tokenizer,
    trace_model,
)
from ..training.logger import setup_logging
from ..training.precision import get_autocast



def save_pickle(base_path, data):
    print('base_path:',base_path)
    os.makedirs(base_path, exist_ok=True)
    file = os.path.join(base_path,'pickle.p')
    print('pickle file location:',file)
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    return file


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def get_dataloader(dataset, batch_size, num_workers):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=None,
    )

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t() #[batch_size, classes] -> [batch_size, 1] -> [1, batch_size], which class # topk = (values, indices)
    correct = pred.eq(target.view(1, -1).expand_as(pred)) #shape: correct=targe.view=pred=[1, batch_size], True or False

    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def run(model, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    with torch.no_grad():
        feature_list = []
        target_list = []
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            target_list.append(target.numpy())
            images = images.to(args.device) #images.shape: torch.Size([batch_size, 3 rgb channels, image_height, image_width])
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)
            

            with autocast():
                image_features = model.encode_image(images) #batch_size x emb_size
                image_features = F.normalize(image_features, dim=-1)
                feature_list.append(image_features.detach().cpu().numpy())
                
        file = save_pickle(log_base_path,[np.vstack(feature_list), np.hstack(target_list), dataloader.dataset.samples,dataloader.dataset.class_to_idx])

    return file



def few_shot_eval(model, data, args):
    results = {}

    logging.info("Starting few-shot.")

    for split in data:
        logging.info("Building few-shot %s classifier.", split)
        
        file = run(model, data[split], args)
        
        logging.info("Finished few-shot %s with total %d classes.", split, len(data[split].dataset.classes))

    logging.info("Finished few-shot.")

    return results, file

def split(select, kshot, nfold, i2c, filepath=None):    
    
    test = [] #N
    target = [] #N
    train = [] #kshot x class
    label = []
    test_sample = []
    random.seed(nfold)
    for k,v in select.items():
        num_v = len(v['feature'])
        if num_v < kshot:
            logging.info(f'{i2c[k]} has only {num_v} images. Less than {kshot} images for few-shot. ')
        elif num_v < kshot+5:
            logging.info(f'{i2c[k]} has only {num_v} images. Not enough for evaluation.')
            
        random.shuffle(v['feature'])
        train.append(v['feature'][:kshot])
        test+=v['feature'][kshot:]
        test_sample+=v['sample'][kshot:]
        label+=[k for i in range(kshot)]
        test_num = num_v-kshot
        target+=[k for i in range(test_num)]
    
    flatten_train = np.vstack(train)
    label = np.array(label)
    assert kshot*len(train) == flatten_train.shape[0] == label.shape[0]
    assert len(train) == n_class
    assert len(test) == len(target) == len(test_sample)

    return flatten_train, label, test, target

def CL2N(x_flatten, x_mean):
    x_flatten = x_flatten - x_mean #(class, emb) = (class, emb) - (emb,)
    x_flatten = x_flatten / LA.norm(x_flatten, 2, 1)[:, None] #(class, emb) = (class, emb) / (class,1)
    return x_flatten

def get_acc(flatten_train, label, test, target, n_class, kshot, nfold):
    train_mean = flatten_train.mean(axis=0)
    
    flatten_train = CL2N(flatten_train,train_mean)
    test = CL2N(test,train_mean)
    train_center = flatten_train.reshape(n_class, kshot, flatten_train.shape[-1]).mean(1)

    num_NN = 1
    label = label[::kshot] #num of class
    subtract = train_center[:, None, :] - test
    distance = LA.norm(subtract, 2, axis=-1) #(train num:test num)
    idx = np.argpartition(distance, num_NN, axis=0)[:num_NN] #(num_NN:train num)
    nearest_samples = np.take(label, idx) #(num_NN:train num)
    out = mode(nearest_samples, axis=0, keepdims=True)[0]
    out = out.astype(int)
    test_label = np.array(target)
    acc = (out == test_label).mean()

    return acc

if __name__ == "__main__":
    global args
    args = parse_args(sys.argv[1:])
    random_seed(args.seed, 0)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = init_device(args)

    args.save_logs = args.logs and args.logs.lower() != "none"

    # get the name of the experiments
    if args.save_logs and args.name is None:
        # sanitize model name for filesystem/uri use
        model_name_safe = args.model.replace("/", "-")
        date_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = "-".join(
            [
                date_str,
                f"model_{model_name_safe}",
                f"b_{args.batch_size}",
                f"p_{args.precision}",
                "few_shot",
            ]
        )
        
    if args.save_logs is None:
        args.log_path = None
    else:
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    if (
        isinstance(args.force_image_size, (tuple, list))
        and len(args.force_image_size) == 1
    ):
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]

    logging.info("Params:")
    if args.save_logs is None:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
    else:
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.task_type == 'eval':
        feature_file = args.pretrained        
    else:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            args.model,
            args.pretrained,
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

        if args.trace:
            model = trace_model(model, batch_size=args.batch_size, device=device)

        logging.info("Model:")
        logging.info(f"{str(model)}")

        # initialize datasets
        data = {
            "val-unseen": get_dataloader(
                DatasetFromFile(args.data_root, args.label_filename, transform=preprocess_val),
                batch_size=args.batch_size,num_workers=args.workers
            ),
        }

        start_time = time.monotonic()
                   
        model.eval()
        _, feature_file = few_shot_eval(model, data, args)             

        end_time = time.monotonic()
        logging.info(f"feature extraction takes: {datetime.timedelta(seconds=end_time - start_time)}")

    if args.task_type == 'eval' or args.task_type == 'all':
        feature, target, samples, c2i = load_pickle(feature_file)

        i2c = dict([(v,k) for k,v in c2i.items()])

        if args.debug:
            for i in range(len(target)):
                assert target[i] == samples[i][1]


        select = dict()
        for idx in range(len(feature)):
            f = feature[idx]
            s = samples[idx]
            cat = target[idx]
            if cat in select:
                select[cat]['feature'].append(f)
                select[cat]['sample'].append(s)
            else:
                select[cat] = dict()
                select[cat]['feature'] = [f]
                select[cat]['sample'] = [s]
                
        count = sum([len(v['feature']) for v in select.values()])
        n_class = len(select)

        logging.info("Num of classes: %d.\nNum of samples: %d.", n_class, count)

        for kshot in args.kshot_list:
           
            acc_list = []
            for n in range(args.nfold):
                #split
                flatten_train, label, test, target = split(select, kshot, n, i2c)
                acc = get_acc(flatten_train, label, test, target, n_class, kshot, n)
                acc_list.append(acc)
                logging.info(f"{kshot} shot No.{n} ACC: {acc:.4f}")
            logging.info("!!!!!!Result: ")
            logging.info("Dataset:  %s", args.data_root)
            logging.info("Model:  %s", args.pretrained)
            logging.info(f"{kshot} shot AVG ACC: {np.mean(acc_list)*100:.2f}")
            logging.info(f"{kshot} shot STD: {np.std(acc_list)*100:.4f}")
    
    logging.info(f"Done!")
