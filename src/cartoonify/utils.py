from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypeAlias, Union
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DiffusionPipeline,
)
from PIL import Image
import csv

import torch.utils


DiffusionOutput: TypeAlias = Tuple[Image.Image, Dict[str, Any]]


def get_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, torch.device):
        return device
    else:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


def init_text_to_image_pipeline(
    device: Optional[Union[str, torch.device]] = None,
) -> StableDiffusionPipeline:
    device = get_device(device)
    repo_id = "lavaman131/cartoonify"
    torch_dtype = torch.float16 if device.type in ["mps", "cuda"] else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(
        repo_id, torch_dtype=torch_dtype
    ).to(device)
    return pipeline


def init_image_to_image_pipeline(
    device: Optional[Union[str, torch.device]] = None,
) -> StableDiffusionImg2ImgPipeline:
    device = get_device(device)
    repo_id = "lavaman131/cartoonify"
    torch_dtype = torch.float16 if device.type in ["mps", "cuda"] else torch.float32
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        repo_id, torch_dtype=torch_dtype
    ).to(device)
    return pipeline


def predict(
    pipeline: DiffusionPipeline,
    pipeline_config: Dict[str, Any],
    device: Optional[Union[str, torch.device]] = None,
    seed: Optional[int] = 42,
) -> DiffusionOutput:
    device = get_device(device)
    # run inference
    if device.type == "cuda":
        with torch.cuda.amp.autocast():
            generator = (
                torch.Generator(device=device).manual_seed(seed) if seed else None
            )
    else:
        generator = torch.Generator(device=device).manual_seed(seed) if seed else None

    image = pipeline(**pipeline_config, generator=generator).images[0]

    metadata = {
        **pipeline_config,
        "seed": seed,
    }

    return image, metadata


def save_image_and_metadata(
    base_dir: Union[str, Path],
    image: Image.Image,
    metadata: Dict[str, Any],
    save_fname: str,
    write_mode: str = "a",
) -> None:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    image_dir = base_dir.joinpath("images")
    image_dir.mkdir(parents=True, exist_ok=True)
    image.save(image_dir.joinpath(save_fname))
    with open(base_dir.joinpath("diffusion_metadata.csv"), mode=write_mode) as f:
        csv_writer = csv.writer(
            f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        # write header if file is empty
        if f.tell() == 0:
            keys = ["image_name"]
            keys += list(metadata.keys())
            csv_writer.writerow(keys)
        values = [save_fname]
        values += list(metadata.values())
        csv_writer.writerow(values)
