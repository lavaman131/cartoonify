#!/bin/bash

python push_model.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" \
    --output_dir "/data/lora-dreambooth-model-v1/checkpoint-9000" \
    --instance_prompt "disney" \
    --train_text_encoder \
    --repo_id "cartoonify" \
    --revision "main" \
    --prior_generation_precision "fp16" \
    --commit_message "training after 9000 steps" \
