import gradio as gr
from functools import partial
from PIL import Image
from cartoonify.utils import init_image_to_image_pipeline, predict as _predict


def preprocess_image(img):
    return img.resize((512, 512))


def generate(
    image, prompt, negative_prompt, random_seed, cfg, denoising, denoising_steps
):
    # init_image = preprocess_image(image)
    image = predict(
        pipeline_config={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "guidance_scale": cfg,
            "strength": denoising,
            "num_inference_steps": denoising_steps,
        },
        seed=random_seed,
    )[0]
    return image


with gr.Blocks() as demo:
    pipeline = init_image_to_image_pipeline()
    predict = partial(_predict, pipeline)
    with gr.Row():
        input_image = gr.Image(type="pil", image_mode="RGB")

        input_image.upload(
            fn=preprocess_image,
            inputs=input_image,
            outputs=input_image,
            trigger_mode="once",
        )
        output_image = gr.Image()

    with gr.Column():
        prompt = gr.Textbox(
            value="disney style person, 8k, ray tracing, detailed",
            label="Prompt (max 77 tokens)",
        )
        negative_prompt = gr.Textbox(
            value="[out of frame], ((noisy)), ((artifacts)), ((blurry)), ((distorted)), ((fake eyes))",
            label="Negative prompt (max 77 tokens)",
        )

        cfg = gr.Slider(
            minimum=2,
            maximum=20,
            value=20,
            step=0.5,
            label="Classifier-free guidance scale (CFG)",
        )
        denoising = gr.Slider(
            minimum=0.0, maximum=1.0, value=0.35, step=0.05, label="Denoising strength"
        )
        denoising_steps = gr.Slider(
            minimum=10, maximum=50, value=50, step=1, label="Denoising steps"
        )
        random_seed = gr.Number(42, label="Random seed")

        with gr.Row():
            run = gr.Button(value="Generate")

            run.click(
                fn=generate,
                inputs=[
                    input_image,
                    prompt,
                    negative_prompt,
                    random_seed,
                    cfg,
                    denoising,
                    denoising_steps,
                ],
                outputs=output_image,
            )
            clear = gr.ClearButton(components=[output_image])


if __name__ == "__main__":
    demo.launch()
