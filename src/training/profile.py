import torch
from torch import nn
import open_clip
import pandas as pd
from fvcore.nn import FlopCountAnalysis, flop_count_str, ActivationCountAnalysis
import fire
from typing import Tuple, Union, Dict, Any
from abc import ABC, abstractmethod


class ModelProtocol(nn.Module, ABC):
    @abstractmethod
    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Any:
        pass

    @property
    @abstractmethod
    def visual(self) -> Any:
        pass

    @property
    @abstractmethod
    def text(self) -> nn.Module:
        pass

    @abstractmethod
    def eval(self) -> None:
        pass

    @abstractmethod
    def cuda(self) -> None:
        pass


def profile_fvcore(
    model: torch.nn.Module,
    image_input_size: Tuple[int, int, int] = (3, 224, 224),
    text_input_size: Tuple[int,] = (77,),
    batch_size: int = 1,
    detailed: bool = False,
    force_cpu: bool = False,
) -> Tuple[float, float]:
    if force_cpu:
        model = model.to("cpu")
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_image_input = torch.ones(
        (batch_size,) + image_input_size, device=device, dtype=dtype
    )
    example_text_input = torch.ones(
        (batch_size,) + text_input_size, device=device, dtype=torch.int64
    )
    fca = FlopCountAnalysis(model, (example_image_input, example_text_input))
    aca = ActivationCountAnalysis(model, (example_image_input, example_text_input))
    if detailed:
        fcs = flop_count_str(fca)
        print(fcs)
    return fca.total(), aca.total()


def profile_fvcore_text(
    model: torch.nn.Module,
    text_input_size: Tuple[int,] = (77,),
    batch_size: int = 1,
    detailed: bool = False,
    force_cpu: bool = False,
) -> Tuple[float, float]:
    if force_cpu:
        model = model.to("cpu")
    device = next(model.parameters()).device
    example_input = torch.ones(
        (batch_size,) + text_input_size, device=device, dtype=torch.int64
    )
    fca = FlopCountAnalysis(model, example_input)
    aca = ActivationCountAnalysis(model, example_input)
    if detailed:
        fcs = flop_count_str(fca)
        print(fcs)
    return fca.total(), aca.total()


def profile_fvcore_image(
    model: torch.nn.Module,
    image_input_size: Tuple[int, int, int] = (3, 224, 224),
    batch_size: int = 1,
    detailed: bool = False,
    force_cpu: bool = False,
) -> Tuple[float, float]:
    if force_cpu:
        model = model.to("cpu")
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.ones(
        (batch_size,) + image_input_size, device=device, dtype=dtype
    )
    fca = FlopCountAnalysis(model, example_input)
    aca = ActivationCountAnalysis(model, example_input)
    if detailed:
        fcs = flop_count_str(fca)
        print(fcs)
    return fca.total(), aca.total()


def count_params(model: torch.nn.Module) -> int:
    return sum([m.numel() for m in model.parameters()])


def profile_model(model_name: str) -> Dict[str, Union[str, int, float]]:
    model = open_clip.create_model(
        model_name, force_custom_text=True, pretrained_hf=False
    )
    assert isinstance(model, ModelProtocol), (
        "Model must be an instance of ModelProtocol"
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    assert hasattr(model, "visual"), "Model must have a 'visual' attribute"
    assert hasattr(model, "text"), "Model must have a 'text' attribute"
    assert model is not None

    visual = model.visual

    assert hasattr(visual, "image_size"), (
        "Model's visual component must have 'image_size'"
    )

    if isinstance(visual.image_size, (tuple, list)):
        image_input_size = (3, int(visual.image_size[-2]), int(visual.image_size[-1]))
    else:
        image_input_size = (3, int(visual.image_size), int(visual.image_size))

    text_input_size = (77,)

    results: Dict[str, Union[str, int, float]] = {}
    results["model"] = model_name
    results["image_size"] = image_input_size[1]  # type: ignore

    model_cfg = open_clip.get_model_config(model_name)
    if model_cfg:
        vision_cfg = open_clip.CLIPVisionCfg(**model_cfg["vision_cfg"])
        text_cfg = open_clip.CLIPTextCfg(**model_cfg["text_cfg"])
        results["image_width"] = int(vision_cfg.width)
        results["text_width"] = int(text_cfg.width)
        results["embed_dim"] = int(model_cfg["embed_dim"])
    else:
        results["image_width"] = 0
        results["text_width"] = 0
        results["embed_dim"] = 0

    retries = 2
    while retries:
        retries -= 1
        try:
            macs, acts = profile_fvcore(
                model,
                image_input_size=image_input_size,
                text_input_size=text_input_size,
                force_cpu=not retries,
            )

            image_macs, image_acts = profile_fvcore_image(
                model.visual, image_input_size=image_input_size, force_cpu=not retries
            )

            text_macs, text_acts = profile_fvcore_text(
                model.text, text_input_size=text_input_size, force_cpu=not retries
            )

            results["gmacs"] = round(macs / 1e9, 2)
            results["macts"] = round(acts / 1e6, 2)
            results["mparams"] = round(count_params(model) / 1e6, 2)
            results["image_gmacs"] = round(image_macs / 1e9, 2)
            results["image_macts"] = round(image_acts / 1e6, 2)
            results["image_mparams"] = round(count_params(model.visual) / 1e6, 2)
            results["text_gmacs"] = round(text_macs / 1e9, 2)
            results["text_macts"] = round(text_acts / 1e6, 2)
            results["text_mparams"] = round(count_params(model.text) / 1e6, 2)
        except RuntimeError as e:
            pass
    return results


def main(model: str = "", results_file: str = ""):
    # FIXME accept a text file name to allow lists of models in txt/csv
    if model == "all":
        parsed_model = open_clip.list_models()
    else:
        parsed_model = model.split(",")

    results = []
    for m in parsed_model:
        row = profile_model(m)
        results.append(row)

    df = pd.DataFrame(results, columns=results[0].keys())  # type: ignore
    df = df.sort_values("gmacs")  # type: ignore
    print(df)
    if results_file:
        df.to_csv(results_file, index=False)


if __name__ == "__main__":
    fire.Fire(main)
