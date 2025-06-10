#!/bin/bash

# Force CPU-only execution
export CUDA_VISIBLE_DEVICES=""

# --- Model Config ---
LLM_VERSION="state-spaces/mamba-130m"  # Smaller model for CPU
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-base-patch16"

RUN_NAME="llava_mamba_cpu_test"

# --- Colab Google Drive Output Directory ---
OUTPUT_DIR="/content/drive/My Drive/llava/checkpoints/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

# --- Training Launch ---
torchrun --nproc-per-node=1 llava/train/train_mem.py \  # Change this if your training script is elsewhere
    --model_name_or_path $LLM_VERSION \
    --version plain \
    --vision_tower $VISION_MODEL_VERSION \
    --mm_projector_type mlp2x_gelu \
    --data_path "./playground/data/llava_instruction_following/llava_instruct_150k.json" \
    --image_folder "./playground/data/llava_instruction_following/images" \
    --tune_mm_mlp_adapter True \
    --mm_use_im_start_end True \
    --use_im_start_end True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 1024 \
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --torch_compile False \
    --dataloader_drop_last False \
    --report_to wandb

exit 0
