from datasets import load_dataset
from argparse import ArgumentParser

def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", help="The local path to the image dataset.")
    parser.add_argument("--repo_id", help="The name of the repo to push to Huggingface Hub.")
    return parser

def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    dataset = load_dataset("./disney_images", data_dir=args.data_dir) 
    dataset.push_to_hub(args.repo_id)


if __name__ == "__main__":
    main()
