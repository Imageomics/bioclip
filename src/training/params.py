from typing import Optional, Union, Tuple
from pydantic import BaseModel, Field, ConfigDict


class Args(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dataset: str = Field(
        default="TreeOfLife",
        alias="dataset",
        description="Name of dataset to use. Options: TreeOfLife, BioTrove, Both",
    )
    train_num_samples: Optional[int] = Field(
        default=None,
        alias="train_num_samples",
        description="Number of samples in training dataset.",
    )
    val_num_samples: Optional[int] = Field(
        default=None,
        alias="val_num_samples",
        description="Number of samples in validation dataset.",
    )
    logs: str = Field(
        default="../storage/model",
        alias="logs",
        description="Where to store tensorboard logs.",
    )
    name: Optional[str] = Field(
        default=None,
        alias="name",
        description="Optional identifier for the experiment when storing logs.",
    )
    workers: int = Field(
        default=8, alias="workers", description="Number of dataloader workers per GPU."
    )
    batch_size: int = Field(
        default=4096, alias="batch_size", description="Batch size per GPU."
    )
    epochs: int = Field(
        default=100, alias="epochs", description="Number of epochs to train for."
    )
    epochs_cooldown: Optional[int] = Field(
        default=None,
        alias="epochs_cooldown",
        description="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards.",
    )
    distill: bool = Field(
        default=False,
        alias="distill",
        description="Whether to distill the model.",
    )
    lr: float = Field(default=1e-4, alias="lr", description="Learning rate.")
    beta1: Optional[float] = Field(
        default=None, alias="beta1", description="Adam beta 1."
    )
    beta2: Optional[float] = Field(
        default=None, alias="beta2", description="Adam beta 2."
    )
    eps: Optional[float] = Field(default=None, alias="eps", description="Adam epsilon.")
    wd: float = Field(default=0.2, alias="wd", description="Weight decay.")
    warmup: int = Field(
        default=1000, alias="warmup", description="Number of steps to warmup for."
    )
    skip_scheduler: bool = Field(
        default=False,
        alias="skip_scheduler",
        description="Use this flag to skip the learning rate decay.",
    )
    lr_scheduler: str = Field(
        default="cosine", alias="lr_scheduler", description="LR scheduler."
    )
    lr_cooldown_end: float = Field(
        default=0.0,
        alias="lr_cooldown_end",
        description="End learning rate for cooldown schedule.",
    )
    lr_cooldown_power: float = Field(
        default=1.0,
        alias="lr_cooldown_power",
        description="Power for polynomial cooldown schedule.",
    )
    save_frequency: int = Field(
        default=1, alias="save_frequency", description="How often to save checkpoints."
    )
    save_most_recent: bool = Field(
        default=False,
        alias="save_most_recent",
        description="Always save the most recent model trained to epoch_latest.pt.",
    )
    zeroshot_frequency: int = Field(
        default=2, alias="zeroshot_frequency", description="How often to run zero shot."
    )
    val_frequency: int = Field(
        default=1,
        alias="val_frequency",
        description="How often to run evaluation with val data.",
    )
    resume: Optional[str] = Field(
        default=None,
        alias="resume",
        description="path to latest checkpoint (default: none)",
    )
    precision: str = Field(
        default="amp", alias="precision", description="Floating point precision."
    )
    model: str = Field(
        default="ViT-B-16",
        alias="model",
        description="Name of the vision backbone to use.",
    )
    pretrained: Optional[str] = Field(
        default=None,
        alias="pretrained",
        description="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    lock_image: bool = Field(
        default=False,
        alias="lock_image",
        description="Lock full image tower by disabling gradients.",
    )
    lock_image_unlocked_groups: int = Field(
        default=0,
        alias="lock_image_unlocked_groups",
        description="Leave last n image tower layer groups unlocked.",
    )
    lock_image_freeze_bn_stats: bool = Field(
        default=False,
        alias="lock_image_freeze_bn_stats",
        description="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    image_mean: Tuple[float, ...] = Field(
        default=(0.485, 0.456, 0.406),
        alias="image_mean",
        description="Override default image mean value of dataset",
    )
    image_std: Tuple[float, ...] = Field(
        default=(0.229, 0.224, 0.225),
        alias="image_std",
        description="Override default image std deviation of of dataset",
    )
    aug_cfg: dict = Field(
        default_factory=dict, alias="aug_cfg", description="augmentation configuration"
    )
    force_image_size: Optional[Union[int, Tuple[int]]] = Field(
        default=None,
        alias="force_image_size",
        description="Override default image size",
    )
    force_quick_gelu: bool = Field(
        default=False,
        alias="force_quick_gelu",
        description="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    force_patch_dropout: Optional[float] = Field(
        default=None,
        alias="force_patch_dropout",
        description="Override the patch dropout during training",
    )
    torchscript: bool = Field(
        default=False, alias="torchscript", description="torch.jit.script the model"
    )
    trace: bool = Field(
        default=False,
        alias="trace",
        description="torch.jit.trace the model for inference / eval only",
    )
    accum_freq: int = Field(
        default=1,
        alias="accum_freq",
        description="Update the model every --acum-freq steps.",
    )
    debug: bool = Field(
        default=False, alias="debug", description="If true, more information is logged."
    )
    seed: int = Field(default=0, alias="seed", description="Default random seed.")
    grad_clip_norm: Optional[float] = Field(
        default=None, alias="grad_clip_norm", description="Gradient clip."
    )
    lock_text: bool = Field(
        default=False,
        alias="lock_text",
        description="Lock full text tower by disabling gradients.",
    )
    lock_text_unlocked_layers: int = Field(
        default=0,
        alias="lock_text_unlocked_layers",
        description="Leave last n image tower layer groups unlocked.",
    )
    lock_text_freeze_layer_norm: bool = Field(
        default=False,
        alias="lock_text_freeze_layer_norm",
        description="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    log_every_n_steps: int = Field(
        default=1,
        alias="log_every_n_steps",
        description="Log every n steps to tensorboard/console/wandb.",
    )
    delete_previous_checkpoint: bool = Field(
        default=False,
        alias="delete_previous_checkpoint",
        description="If true, delete previous checkpoint after storing a new one.",
    )
    distill_model: Optional[str] = Field(
        default=None,
        alias="distill_model",
        description="Which model arch to distill from, if any.",
    )
    distill_pretrained: Optional[str] = Field(
        default=None,
        alias="distill_pretrained",
        description="Which pre-trained weights to distill from, if any.",
    )
    text_type: str = Field(
        default="taxon",
        alias="text_type",
        description="Text type of annotation for text encoder.",
    )
    device: str = Field(default="cuda", alias="device", description="Torch device")
    start_epoch: int = Field(default=0, alias="start_epoch", description="Start epoch")
    save_logs: bool = Field(default=True, alias="save_logs", description="Save logs")
    tensorboard: bool = Field(
        default=False, alias="tensorboard", description="Use tensorboard"
    )
    log_path: Optional[str] = Field(
        default=None, alias="log_path", description="Log path"
    )
    checkpoint_path: str = Field(
        default="", alias="checkpoint_path", description="Checkpoint path"
    )
    tensorboard_path: str = Field(
        default="", alias="tensorboard_path", description="Tensorboard path"
    )
    log_level: int = Field(default=0, alias="log_level", description="Log level")
    rank: int = Field(default=0, alias="rank", description="Rank")


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
