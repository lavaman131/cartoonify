from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm


def main() -> None:
    DATA_DIR = Path("../resized_disney")
    SAVE_DIR = Path("../disney_processed")
    SAVE_DIR.mkdir(exist_ok=True)
    images = list(DATA_DIR.glob("*.png"))
    for idx, img in enumerate(tqdm(images)):
        img = Image.open(img)
        img.save(SAVE_DIR.joinpath(f"{idx}.png"))


if __name__ == "__main__":
    main()
