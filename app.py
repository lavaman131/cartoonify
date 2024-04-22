import torch
import gradio as gr
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float32).to("mps")
pipeline.load_lora_weights("lavaman131/cartoonify", weight_name="pytorch_lora_weights.safetensors")

image = pipeline("disney style cute baby lion", negative_prompt="person human", num_inference_steps=30, guidance_scale=7.5, seed=42).images[0]
image.save("baby_lion.png")
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

# if __name__ == "__main__":
#     demo.launch()
