from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypeAlias, Union
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from PIL import Image
import csv

from cartoonify.utils import predict, save_image_and_metadata

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
    pass
    # demo.launch()
