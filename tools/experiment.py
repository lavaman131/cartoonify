from pathlib import Path
from cartoonify.utils import predict, save_image_and_metadata


if __name__ == "__main__":
    # image, metadata = predict(
    #     {
    #         "prompt": "a man wearing a green hat, disney style",
    #         "negative_prompt": "",
    #         "guidance_scale": 8,
    #         "num_inference_steps": 30,
    #     },
    #     seed=42,
    #     device="mps",
    # )
    # base_dir = Path("../examples")
    # save_fname = "green_hat_man.png"
    # save_image_and_metadata(base_dir, image, metadata, save_fname)

    image, metadata = predict(
        {
            "prompt": "disney style (baby lion)",
            "negative_prompt": "person human",
            "guidance_scale": 8,
            "num_inference_steps": 30,
        },
        seed=42,
        device="mps",
    )
    base_dir = Path("../examples")
    save_fname = "baby_lion.png"
    save_image_and_metadata(base_dir, image, metadata, save_fname)

    # image, metadata = predict(
    #     {
    #         "prompt": "disney style legend of zelda riding on a brown horse",
    #         "negative_prompt": "",
    #         "guidance_scale": 8.0,
    #         "num_inference_steps": 30,
    #     },
    #     seed=42,
    #     device="mps",
    # )
    # base_dir = Path("../examples")
    # save_fname = "legend_of_zelda.png"
    # save_image_and_metadata(base_dir, image, metadata, save_fname)
