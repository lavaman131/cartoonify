from argparse import ArgumentParser
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--local_dir", help="The local path to save dataset from Huggingface Hub."
    )
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    dataset = load_dataset("ywnl/disney_images", split="train")
    save_dir = Path(args.local_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(tqdm(dataset["image"])):
        img.save(save_dir / f"{idx}.png")


if __name__ == "__main__":
    main()
