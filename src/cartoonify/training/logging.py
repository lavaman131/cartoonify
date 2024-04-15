import importlib
from typing import Any, Dict, List, Optional, Union

import numpy as np
from omegaconf import DictConfig
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from PIL import Image

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.import_utils import is_wandb_available
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from accelerate.logging import MultiProcessAdapter
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_utils import PreTrainedModel
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)

if is_wandb_available():
    import wandb


def log_validation_lora(
    config: DictConfig,
    logger: MultiProcessAdapter,
    pipeline: DiffusionPipeline,
    accelerator: Accelerator,
    pipeline_config: Dict[str, Any],
    epoch: int,
    is_final_validation: bool = False,
) -> List[Image.Image]:
    logger.info(
        f"Running validation... \n Generating {config.num_validation_images} images with prompt:"
        f" {config.validation_prompt}."
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_config = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_config["variance_type"] = variance_type

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config, **scheduler_config
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = (
        torch.Generator(device=accelerator.device).manual_seed(config.seed)
        if config.seed
        else None
    )

    if config.validation_images is None:
        images = []
        for _ in range(config.num_validation_images):
            with torch.cuda.amp.autocast():
                image = pipeline(**pipeline_config, generator=generator).images[0]  # type: ignore
                images.append(image)
    else:
        images = []
        for image in config.validation_images:
            image = Image.open(image)
            with torch.cuda.amp.autocast():
                image = pipeline(
                    **pipeline_config, image=image, generator=generator
                ).images[0]  # type: ignore
            images.append(image)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {config.validation_prompt}")
                        for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()

    return images


def log_validation(
    config: DictConfig,
    logger: MultiProcessAdapter,
    text_encoder: Optional[PreTrainedModel],
    tokenizer: Union[None, PreTrainedTokenizer, PreTrainedTokenizerFast],
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    global_step: int,
    prompt_embeds: Optional[torch.Tensor],
    negative_prompt_embeds: Optional[torch.Tensor],
) -> List[Image.Image]:
    logger.info(
        f"Running validation... \n Generating {config.num_validation_images} images with prompt:"
        f" {config.validation_prompt}."
    )

    pipeline_config = {}

    if vae is not None:
        pipeline_config["vae"] = vae

    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        revision=config.revision,
        variant=config.variant,
        torch_dtype=weight_dtype,
        **pipeline_config,
    )

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_config = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_config["variance_type"] = variance_type

    module = importlib.import_module("diffusers")
    scheduler_class = getattr(module, config.validation_scheduler)
    pipeline.scheduler = scheduler_class.from_config(
        pipeline.scheduler.config, **scheduler_config
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if config.pre_compute_text_embeddings:
        pipeline_config = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }
    else:
        pipeline_config = {"prompt": config.validation_prompt}

    # run inference
    generator = (
        None
        if config.seed is None
        else torch.Generator(device=accelerator.device).manual_seed(config.seed)
    )
    images = []
    if config.validation_images is None:
        for _ in range(config.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    **pipeline_config, num_inference_steps=25, generator=generator
                ).images[0]
            images.append(image)
    else:
        for image in config.validation_images:
            image = Image.open(image)
            image = pipeline(
                **pipeline_config, image=image, generator=generator
            ).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "validation", np_images, global_step, dataformats="NHWC"
            )
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {config.validation_prompt}")
                        for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()

    return images
