from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypeAlias, Union
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from PIL import Image
import csv


@dataclass
class DiffusionMetadata:
    prompt: str
    negative_prompt: str
    guidance_scale: int
    num_inference_steps: int
    seed: int


DiffusionOutput: TypeAlias = Tuple[Image.Image, DiffusionMetadata]


def predict(
    pipeline_config: Dict[str, Any],
    device: Union[str, torch.device],
    seed: Optional[int] = 42,
) -> DiffusionOutput:
    device = torch.device(device) if isinstance(device, str) else device
    repo_id = "lavaman131/cartoonify"
    pipeline = StableDiffusionPipeline.from_pretrained(
        repo_id, torch_dtype=torch.float16
    ).to(device)

    # run inference
    if device.type == "cuda":
        with torch.cuda.amp.autocast():
            generator = (
                torch.Generator(device=device).manual_seed(seed) if seed else None
            )
    else:
        generator = torch.Generator(device=device).manual_seed(seed) if seed else None

    image = pipeline(**pipeline_config, generator=generator).images[0]

    metadata_kwargs = {
        "prompt": pipeline_config["prompt"],
        "negative_prompt": pipeline_config["negative_prompt"],
        "guidance_scale": pipeline_config["guidance_scale"],
        "num_inference_steps": pipeline_config["num_inference_steps"],
        "seed": seed,
    }

    metadata = DiffusionMetadata(**metadata_kwargs)

    return image, metadata


def save_image_and_metadata(
    base_dir: Union[str, Path],
    image: Image.Image,
    metadata: DiffusionMetadata,
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
            keys += list(metadata.__dict__.keys())
            csv_writer.writerow(keys)
        values = [save_fname]
        values += list(metadata.__dict__.values())
        csv_writer.writerow(values)
