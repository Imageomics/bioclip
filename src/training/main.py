import glob
import logging
import os
import re
import numpy as np
import random
from datetime import datetime
from typing import (
    Protocol,
    runtime_checkable,
    Any,
    Optional,
    Tuple,
    Dict,
    Callable,
    List,
)
import fire
import torch
from torch import optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter


from open_clip import (
    create_model_and_transforms,
    get_tokenizer,
    create_loss,
)
from ..training.data import get_data, DataInfo
from ..training.logger import setup_logging
from ..training.params import Args
from ..training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from ..training.train import train_one_epoch, evaluate


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


@runtime_checkable
class ModelProtocol(Protocol):
    """A protocol defining the expected interface for the model."""

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state_dict: dict) -> Any: ...

    def set_grad_checkpointing(self) -> Any: ...

    def named_parameters(self) -> Any: ...


@runtime_checkable
class DataLoaderProtocol(Protocol):
    """A protocol defining the expected interface for the dataloader."""

    @property
    def num_batches(self) -> int: ...


def random_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def natural_key(string_: str) -> List[Any]:
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def get_latest_checkpoint(path: str) -> Optional[str]:
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    checkpoints = glob.glob(path + "**/*.pt", recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def setup_experiment(args: Args) -> Args:
    """Sets up the experiment environment, including logging, tensorboard, and checkpoint paths."""
    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace("/", "-")
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        args.name = "-".join(
            [
                s
                for s in [
                    date_str,
                    f"model_{model_name_safe}",
                    f"lr_{args.lr}",
                    f"b_{args.batch_size}",
                    f"j_{args.workers}",
                    f"p_{args.precision}",
                ]
                if s is not None
            ]
        )

    resume_latest = args.resume == "latest"
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    os.makedirs(log_base_path, exist_ok=True)
    log_filename = "out.log"
    args.log_path = os.path.join(log_base_path, log_filename)
    if os.path.exists(args.log_path) and not resume_latest:
        print(
            "Error. Experiment already exists. Use --name {} to specify a new experiment."
        )
        return args

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    args.tensorboard_path = (
        os.path.join(log_base_path, "tensorboard") if args.tensorboard else ""
    )
    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    return args


def handle_resume(args: Args) -> Args:
    """Handles resuming from a checkpoint, including finding the latest checkpoint."""
    resume_latest = args.resume == "latest"
    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        if args.save_most_recent:
            # if --save-most-recent flag is set, look for latest at a fixed filename
            resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
            if not os.path.exists(resume_from):
                # If no latest checkpoint has been saved yet, don't try to resume
                resume_from = None
        else:
            # otherwise, list checkpoint dir contents and pick the newest checkpoint
            resume_from = get_latest_checkpoint(checkpoint_path)
        if resume_from:
            logging.info(f"Found latest resume checkpoint at {resume_from}.")
        else:
            logging.info(f"No latest resume checkpoint found in {checkpoint_path}.")
        args.resume = resume_from
    return args


def _check_model_requirements(model: nn.Module) -> bool:
    """Check if model has all required methods."""
    required_methods = ["state_dict", "load_state_dict", "set_grad_checkpointing"]
    return all(hasattr(model, method) for method in required_methods)


def initialize_model(
    args: Args, device: torch.device
) -> Tuple[
    Any,
    Any,
    Any,
    Any,
    Any,
    Any,
]:
    """Creates and initializes the model and optimizer."""
    dist_model: Optional[nn.Module] = None
    args.distill = (
        args.distill_model is not None and args.distill_pretrained is not None
    )
    if args.distill:
        # FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        # FIXME: support distillation with coca.
        assert "coca" not in args.model.lower()

    if (
        isinstance(args.force_image_size, (tuple, list))
        and len(args.force_image_size) == 1
    ):
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = int(args.force_image_size[0])
    random_seed(args.seed)

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
    )
    if args.distill:
        assert args.distill_model is not None and args.distill_pretrained is not None
        # FIXME: currenlty assumes the model your distilling from has the same tokenizer & transforms.
        dist_model_, _, _ = create_model_and_transforms(
            args.distill_model,
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
        )
        assert isinstance(dist_model_, nn.Module)
        dist_model = dist_model_
    else:
        dist_model = None

    if not isinstance(model, ModelProtocol):  # type: ignore
        raise TypeError(
            "Model does not conform to ModelProtocol. "
            "Ensure it has forward(), state_dict(), load_state_dict(), and set_grad_checkpointing() methods."
        )

    logging.info("Model:")
    logging.info(f"{str(model)}")
    logging.info("Params:")
    assert args.name is not None
    params_file = os.path.join(args.logs, args.name, "params.txt")
    with open(params_file, "w") as f:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
            f.write(f"{name}: {val}\n")

    # create optimizer and scaler
    optimizer: Optional[Optimizer] = None
    scaler: Optional[GradScaler] = None

    assert args.beta1 is not None
    assert args.beta2 is not None
    assert args.eps is not None

    def exclude(n: str, p: torch.Tensor) -> bool:
        return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n

    def include(n: str, p: torch.Tensor) -> bool:
        return not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad
    ]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    scaler = GradScaler() if args.precision == "amp" else None
    return model, preprocess_train, preprocess_val, optimizer, scaler, dist_model


def initialize_data(
    args: Args, preprocess_train: Callable, preprocess_val: Callable
) -> Dict[str, DataInfo]:
    """Initializes datasets."""
    tokenizer = get_tokenizer(args.model)
    seed = args.seed
    train_data = get_data(
        seed=seed,
        dataset=args.dataset,
        tokenizer=tokenizer,
        transform=preprocess_train,
        split="train",
        batch_size=args.batch_size,
    )
    val_data = get_data(
        seed=seed,
        dataset=args.dataset,
        tokenizer=tokenizer,
        transform=preprocess_val,
        split="val",
        batch_size=args.batch_size,
    )
    data = {"train": train_data, "val": val_data}
    assert all(isinstance(v, DataInfo) for v in data.values())
    return data


def setup_scheduler(optimizer: Optimizer, data: Dict[str, DataInfo], args: Args) -> Any:
    """Sets up the learning rate scheduler."""
    scheduler = None
    if "train" in data and optimizer is not None:
        if not isinstance(data["train"].dataloader, DataLoaderProtocol):
            raise TypeError(
                "DataLoader does not conform to DataLoaderProtocol. "
                "Ensure it has a num_batches property."
            )
        total_steps = (
            data["train"].dataloader.num_batches // args.accum_freq
        ) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None, (
                "Please specify the number of cooldown epochs for this lr schedule."
            )
            cooldown_steps = (
                data["train"].dataloader.num_batches // args.accum_freq
            ) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer,
                args.lr,
                args.warmup,
                total_steps,
                cooldown_steps,
                args.lr_cooldown_power,
                args.lr_cooldown_end,
            )
        else:
            logging.error(
                f"Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown."
            )
            exit(1)
    return scheduler


def load_checkpoint(
    model: nn.Module, optimizer: Optimizer, scaler: GradScaler, args: Args
) -> int:
    """Loads a checkpoint and updates model, optimizer, and scaler states."""
    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "epoch" in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if next(iter(sd.items()))[0].startswith("module"):
                sd = {k[len("module.") :]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
            logging.info(
                f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})"
            )
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
    return start_epoch


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    completed_epoch: int,
    args: Args,
) -> None:
    """Saves a checkpoint."""
    if args.save_logs:
        checkpoint_dict = {
            "epoch": completed_epoch,
            "name": args.name,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if scaler is not None:
            checkpoint_dict["scaler"] = scaler.state_dict()

        if completed_epoch == args.epochs or (
            args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
        ):
            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
            )
        if args.delete_previous_checkpoint:
            previous_checkpoint = os.path.join(
                args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt"
            )
            if os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)

        if args.save_most_recent:
            # try not to corrupt the latest checkpoint if save fails
            tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
            latest_save_path = os.path.join(
                args.checkpoint_path, LATEST_CHECKPOINT_NAME
            )
            torch.save(checkpoint_dict, tmp_save_path)
            os.replace(tmp_save_path, latest_save_path)


def train_and_evaluate(
    model: nn.Module,
    data: Dict[str, DataInfo],
    optimizer: Optimizer,
    scaler: GradScaler,
    scheduler: Any,
    dist_model: Optional[nn.Module],
    loss: Callable,
    args: Args,
    writer: Optional[SummaryWriter],
) -> None:
    """Encapsulates the training and evaluation loop."""
    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f"Start epoch {epoch}")

        train_one_epoch(
            model,
            data,
            loss,
            epoch,
            optimizer,
            scaler,
            scheduler,
            dist_model,
            args,
            tb_writer=writer,
        )
        completed_epoch = epoch + 1

        if any(v in data for v in ("val", "imagenet-val", "imagenet-v2")):
            evaluate(model, data, completed_epoch, args, writer)

        # Saving checkpoints.
        save_checkpoint(model, optimizer, scaler, completed_epoch, args)


def main(
    dataset: str = "TreeOfLife",
    train_num_samples: Optional[int] = None,
    val_num_samples: Optional[int] = None,
    logs: str = "../storage/model",
    name: Optional[str] = None,
    workers: int = 8,
    batch_size: int = 4096,
    epochs: int = 100,
    epochs_cooldown: Optional[int] = None,
    distill: bool = False,
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.998,
    eps: float = 1.0e-6,
    wd: float = 0.2,
    warmup: int = 1000,
    skip_scheduler: bool = False,
    lr_scheduler: str = "cosine",
    lr_cooldown_end: float = 0.0,
    lr_cooldown_power: float = 1.0,
    save_frequency: int = 1,
    save_most_recent: bool = False,
    zeroshot_frequency: int = 2,
    val_frequency: int = 1,
    resume: Optional[str] = None,
    precision: str = "amp",
    model: str = "ViT-B-16",
    pretrained: Optional[str] = None,
    lock_image: bool = False,
    lock_image_unlocked_groups: int = 0,
    lock_image_freeze_bn_stats: bool = False,
    image_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    image_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    aug_cfg: dict = {},
    force_image_size: Optional[int | Tuple[int]] = None,
    force_quick_gelu: bool = False,
    force_patch_dropout: Optional[float] = None,
    torchscript: bool = False,
    trace: bool = False,
    accum_freq: int = 1,
    debug: bool = False,
    seed: int = 0,
    grad_clip_norm: Optional[float] = None,
    lock_text: bool = False,
    lock_text_unlocked_layers: int = 0,
    lock_text_freeze_layer_norm: bool = False,
    log_every_n_steps: int = 1,
    delete_previous_checkpoint: bool = False,
    distill_model: Optional[str] = None,
    distill_pretrained: Optional[str] = None,
    text_type: str = "taxon",
    device: str = "cuda",
    start_epoch: int = 0,
    save_logs: bool = True,
    tensorboard: bool = False,
    log_path: Optional[str] = None,
    checkpoint_path: str = "",
    tensorboard_path: str = "",
    log_level: int = 0,
    rank: int = 0,
) -> None:
    args = Args(
        dataset=dataset,
        train_num_samples=train_num_samples,
        val_num_samples=val_num_samples,
        logs=logs,
        name=name,
        workers=workers,
        batch_size=batch_size,
        epochs=epochs,
        epochs_cooldown=epochs_cooldown,
        distill=distill,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        wd=wd,
        warmup=warmup,
        skip_scheduler=skip_scheduler,
        lr_scheduler=lr_scheduler,
        lr_cooldown_end=lr_cooldown_end,
        lr_cooldown_power=lr_cooldown_power,
        save_frequency=save_frequency,
        save_most_recent=save_most_recent,
        zeroshot_frequency=zeroshot_frequency,
        val_frequency=val_frequency,
        resume=resume,
        precision=precision,
        model=model,
        pretrained=pretrained,
        lock_image=lock_image,
        lock_image_unlocked_groups=lock_image_unlocked_groups,
        lock_image_freeze_bn_stats=lock_image_freeze_bn_stats,
        image_mean=image_mean,
        image_std=image_std,
        aug_cfg=aug_cfg,
        force_image_size=force_image_size,
        force_quick_gelu=force_quick_gelu,
        force_patch_dropout=force_patch_dropout,
        torchscript=torchscript,
        trace=trace,
        accum_freq=accum_freq,
        debug=debug,
        seed=seed,
        grad_clip_norm=grad_clip_norm,
        lock_text=lock_text,
        lock_text_unlocked_layers=lock_text_unlocked_layers,
        lock_text_freeze_layer_norm=lock_text_freeze_layer_norm,
        log_every_n_steps=log_every_n_steps,
        delete_previous_checkpoint=delete_previous_checkpoint,
        distill_model=distill_model,
        distill_pretrained=distill_pretrained,
        text_type=text_type,
        device=device,
        start_epoch=start_epoch,
        save_logs=save_logs,
        tensorboard=tensorboard,
        log_path=log_path,
        checkpoint_path=checkpoint_path,
        tensorboard_path=tensorboard_path,
        log_level=log_level,
        rank=rank,
    )

    # fully initialize device environment
    device_ = torch.device(args.device)

    # Setup experiment
    args = setup_experiment(args)
    if args == -1:
        return

    # Handle resume
    args = handle_resume(args)
    if args is None:
        return

    # Initialize model
    model_, preprocess_train, preprocess_val, optimizer, scaler, dist_model = (
        initialize_model(args, device_)
    )

    # Load checkpoint
    args.start_epoch = load_checkpoint(model_, optimizer, scaler, args)

    # Initialize datasets
    data = initialize_data(args, preprocess_train, preprocess_val)

    # Create scheduler if train
    scheduler = setup_scheduler(optimizer, data, args)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs is not None and args.logs.lower() != "none"
    writer: Optional[SummaryWriter] = None
    if args.save_logs and args.tensorboard:
        writer = tensorboard.SummaryWriter(args.tensorboard_path)  # type: ignore

    if "train" not in data:
        evaluate(model_, data, args.start_epoch, args, writer)
        return

    loss = create_loss(args)

    train_and_evaluate(
        model_, data, optimizer, scaler, scheduler, dist_model, loss, args, writer
    )


if __name__ == "__main__":
    fire.Fire(main)
