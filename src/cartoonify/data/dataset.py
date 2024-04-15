import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from omegaconf import DictConfig
from transformers import BatchEncoding
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose
from huggingface_hub import model_info
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_utils import PreTrainedModel


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
        class_data_root: Optional[str] = None,
        class_prompt: Optional[str] = None,
        class_num: Optional[int] = None,
        size: int = 512,
        center_crop: bool = False,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        class_prompt_encoder_hidden_states: Optional[torch.Tensor] = None,
        tokenizer_max_length: Optional[int] = None,
    ) -> None:
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(
                f"Instance {self.instance_data_root} images root doesn't exists."
            )

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":  # type: ignore
            instance_image = instance_image.convert("RGB")  # type: ignore
        example["instance_images"] = self.image_transforms(instance_image)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer,
                self.instance_prompt,
                tokenizer_max_length=self.tokenizer_max_length,
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":  # type: ignore
                class_image = class_image.convert("RGB")  # type: ignore
            example["class_images"] = self.image_transforms(class_image)

            if self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer,
                    self.class_prompt,  # type: ignore
                    tokenizer_max_length=self.tokenizer_max_length,
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

        return example


def collate_fn(
    examples: List[Dict[str, torch.Tensor]], with_prior_preservation: bool = False
) -> Dict[str, torch.Tensor]:
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

        if has_attention_mask:
            attention_mask.extend(
                [example["class_attention_mask"] for example in examples]
            )

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if has_attention_mask:
        attention_mask = torch.cat(attention_mask, dim=0)
        batch["attention_mask"] = attention_mask

    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt: str, num_samples: int) -> None:
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, Union[str, int]]:
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def model_has_vae(config: DictConfig) -> bool:
    config_file_name = os.path.join("vae", AutoencoderKL.config_name)
    if os.path.isdir(config.pretrained_model_name_or_path):
        config_file_name = os.path.join(
            config.pretrained_model_name_or_path, config_file_name
        )
        return os.path.isfile(config_file_name)
    else:
        files_in_repo = model_info(
            config.pretrained_model_name_or_path, revision=config.revision
        ).siblings
        return any(file.rfilename == config_file_name for file in files_in_repo)  # type: ignore


def tokenize_prompt(
    tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
    prompt: str,
    tokenizer_max_length: Optional[int] = None,
) -> BatchEncoding:
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(
    text_encoder: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    text_encoder_use_attention_mask: bool = False,
) -> torch.Tensor:
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask and attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds
