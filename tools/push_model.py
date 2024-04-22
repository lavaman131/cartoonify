from argparse import ArgumentParser
import os
from cartoonify.training.utils import save_model_card
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from huggingface_hub import create_repo, upload_folder
import torch


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="The pretrained model name or path.",
    )
    parser.add_argument("--prior_generation_precision", type=str, default="fp32", help="The precision of the prior generation.")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory to save the model card.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        help="The instance prompt to train the model.",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder.",
    )
    parser.add_argument("--repo_id", type=str, help="The repo id.")
    parser.add_argument(
        "--revision",
        default=None,
        type=str,
        help="The revision of the model.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        type=str,
        help="The variant of the model.",
    )
    return parser

def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    repo_id = create_repo(
                repo_id=args.repo_id,
                exist_ok=True,
                token=os.environ["HF_TOKEN"],
            ).repo_id
    
    if args.prior_generation_precision == "fp32":
        torch_dtype = torch.float32
    elif args.prior_generation_precision == "fp16":
        torch_dtype = torch.float16
    elif args.prior_generation_precision == "bf16":
        torch_dtype = torch.bfloat16
    pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    revision=args.revision,
                    variant=args.variant,
                )

    save_model_card(
        repo_id=repo_id,
        repo_folder=args.output_dir,
        prompt="disney style",
        base_model=args.pretrained_model_name_or_path,
        images=None,
        train_text_encoder=args.train_text_encoder,
        pipeline=pipeline)
    
    upload_folder(
        repo_id=repo_id,
        folder_path=args.output_dir,
        commit_message="10000 steps of training",
        ignore_patterns=["step_*", "epoch_*"],
    )


if __name__ == "__main__":
    main()
