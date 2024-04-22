from typing import Any, Dict, Optional, Union
import torch
import gradio as gr
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers import StableDiffusionPipeline
import torch
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
)
from PIL import Image


def predict(
    pipeline_config: Dict[str, Any],
    device: Union[str, torch.device],
    seed: Optional[int] = 42,
) -> Image.Image:
    device = torch.device(device) if isinstance(device, str) else device
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model_id = "nitrosocke/mo-di-diffusion"
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    # pipeline = StableDiffusionPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch_dtype
    # ).to(device)
    # pipeline.load_lora_weights(
    #     "lavaman131/cartoonify", weight_name="pytorch_lora_weights.safetensors"
    # )

    scheduler_config = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_config["variance_type"] = variance_type

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, **scheduler_config
    )

    # run inference
    if device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=torch_dtype):
            generator = (
                torch.Generator(device=device).manual_seed(seed) if seed else None
            )
    else:
        generator = torch.Generator(device=device).manual_seed(seed) if seed else None

    image = pipeline(**pipeline_config, generator=generator).images[0]

    return image


# # move to GPU if available
# if torch.cuda.is_available():
#     generator = generator.to("cuda")

# def generate(prompts):
#   images = generator(list(prompts)).images
#   return [images]

# demo = gr.Interface(generate,
#              "textbox",
#              "image",
#              batch=True,
#              max_batch_size=4  # Set the batch size based on your CPU/GPU memory
# ).queue()

if __name__ == "__main__":
    image = predict(
        {
            "prompt": "a man wearing a green hat, disney style",
            "guidance_scale": 8,
            "num_inference_steps": 30,
        },
        device="mps",
    )
    image.save("lion.png")
    # demo.launch()
