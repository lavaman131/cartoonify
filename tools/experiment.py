from pathlib import Path
from cartoonify.utils import predict, save_image_and_metadata


if __name__ == "__main__":
    image, metadata = predict(
        {
            "prompt": "a close-up of a man with brown skin wearing a golden crown with emerald jewels, disney style, 8K, lens flare",
            "negative_prompt": "((disfigured)), ((bad art)), ((deformed)), ((poorly drawn)), ((extra limbs)), ((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame]",
            "guidance_scale": 8,
            "num_inference_steps": 30,
        },
        seed=42,
        device="mps",
    )
    base_dir = Path("../examples")
    save_fname = "hat_man.png"
    save_image_and_metadata(base_dir, image, metadata, save_fname)

    # image, metadata = predict(
    #     {
    #         "prompt": "disney style (baby lion)",
    #         "negative_prompt": "person human",
    #         "guidance_scale": 8,
    #         "num_inference_steps": 30,
    #     },
    #     seed=42,
    #     device="mps",
    # )
    # base_dir = Path("../examples")
    # save_fname = "baby_lion.png"
    # save_image_and_metadata(base_dir, image, metadata, save_fname)

    # image, metadata = predict(
    #     {
    #         "prompt": "disney style, man wearing shining metalic armor carries a magical sword and shield riding on a brown horse, high resolution, 8k, beautiful colors, detailed, realistic",
    #         "negative_prompt": "((disfigured)), ((bad art)), ((deformed)), ((poorly drawn)), ((extra limbs)), ((close up)), ((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame]",
    #         "guidance_scale": 8.0,
    #         "num_inference_steps": 30,
    #     },
    #     seed=42,
    #     device="mps",
    # )
    # base_dir = Path("../examples")
    # save_fname = "legend_of_zelda.png"
    # save_image_and_metadata(base_dir, image, metadata, save_fname)

    # image, metadata = predict(
    #     {
    #         "prompt": "disney style red ferrari car, high resolution, 8k, beautiful colors, detailed",
    #         "negative_prompt": "((disfigured)), ((bad art)), ((deformed)), ((poorly drawn)), weird colors, blurry, (((duplicate)))",
    #         "guidance_scale": 8,
    #         "num_inference_steps": 30,
    #     },
    #     seed=42,
    #     device="mps",
    # )
    # base_dir = Path("../examples")
    # save_fname = "red_ferrari.png"
    # save_image_and_metadata(base_dir, image, metadata, save_fname)

    # image, metadata = predict(
    #     {
    #         "prompt": "disney style, close-up magical princess, high resolution, 8k, beautiful colors, detailed",
    #         "negative_prompt": "((disfigured)), ((bad art)), ((deformed)), ((poorly drawn)), ((extra limbs)), ((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], ((noisy))",
    #         "guidance_scale": 8,
    #         "num_inference_steps": 30,
    #     },
    #     seed=42,
    #     device="mps",
    # )
    # base_dir = Path("../examples")
    # save_fname = "princess.png"
    # save_image_and_metadata(base_dir, image, metadata, save_fname)

    image, metadata = predict(
        {
            "prompt": "a disney style, close-up black pony, high resolution, 8k, beautiful colors, detailed",
            "negative_prompt": "((disfigured)), ((bad art)), ((deformed)), ((poorly drawn)), ((extra limbs)), ((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], ((noisy))",
            "guidance_scale": 8,
            "num_inference_steps": 30,
        },
        seed=42,
        device="mps",
    )
    base_dir = Path("../examples")
    save_fname = "pony.png"
    save_image_and_metadata(base_dir, image, metadata, save_fname)
