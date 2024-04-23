import torch
import gradio as gr
from functools import partial
from PIL import Image
from cartoonify.utils import init_image_to_image_pipeline, predict as _predict

pipeline = init_image_to_image_pipeline()
predict = partial(_predict, pipeline)

prompt = gr.Textbox(
    value="disney style, beautiful, detailed", label="Prompt (max 77 tokens)"
)
negative_prompt = gr.Textbox(
    value="",
    label="Negative prompt (max 77 tokens)",
)

cfg = gr.Slider(
    minimum=2,
    maximum=15,
    value=7.5,
    step=0.5,
    label="Classifier-free guidance scale (CFG)",
)
denoising = gr.Slider(
    minimum=0.0, maximum=1.0, value=0.75, step=0.05, label="Denoising strength"
)
denoising_steps = gr.Slider(
    minimum=10, maximum=50, value=50, step=1, label="Denoising steps"
)
random_seed = gr.Number(42, label="Random seed")


def generate(
    image, prompt, negative_prompt, random_seed, cfg, denoising, denoising_steps
):
    init_image = Image.open(image).convert("RGB")
    image = predict(
        pipeline_config={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": init_image,
            "guidance_scale": cfg,
            "strength": denoising,
            "num_inference_steps": denoising_steps,
            "height": 512,
            "width": 512,
            "resize_mode": "fill",
        },
        seed=random_seed,
    )[0]
    return image


demo = gr.Interface(
    generate,
    inputs=[
        gr.Image(type="filepath"),
        prompt,
        negative_prompt,
        random_seed,
        cfg,
        denoising,
        denoising_steps,
    ],
    outputs=gr.Image(width=512, height=512),
).queue()

if __name__ == "__main__":
    demo.launch()
