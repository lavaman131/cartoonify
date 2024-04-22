#!/bin/bash

python push_model.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" \
    --output_dir "lora-dreambooth-model/checkpoint-10000" \
    --instance_prompt "disney style" \
    --train_text_encoder \
    --repo_id "cartoonify" \
    --prior_generation_precision "fp32" \
