import os
from typing import List, Optional, Type, Union
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from PIL import Image
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
import numpy as np
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig


def save_model_card(
    repo_id: str,
    repo_folder: str,
    prompt: Optional[str],
    base_model: Optional[str],
    images: Union[None, List[Image.Image], np.ndarray] = None,
    train_text_encoder: bool = False,
    pipeline: Optional[DiffusionPipeline] = None,
) -> None:
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# DreamBooth - {repo_id}

This is a dreambooth model derived from {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/).
You can find some example images in the following. \n
{img_str}

DreamBooth for the text encoder was enabled: {train_text_encoder}.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,  # type: ignore
        prompt=prompt,
        model_description=model_description,
        inference=True,
    )

    tags = ["text-to-image", "dreambooth", "diffusers-training"]
    if isinstance(pipeline, StableDiffusionPipeline):
        tags.extend(["stable-diffusion", "stable-diffusion-diffusers"])
    else:
        tags.extend(["if", "if-diffusers"])
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
) -> Type[PreTrainedModel]:
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.deprecated.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")
