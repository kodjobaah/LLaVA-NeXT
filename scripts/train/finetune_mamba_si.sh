#!/bin/bash

# --- Colab Specific Setup ---
# In Colab, you usually don't need to manually set these NCCL variables unless you're
# trying to do multi-node training, which is complex and expensive in Colab.
# For single-node, single-GPU, they are often redundant or even problematic.
# If you get a multi-GPU A100 instance (rare for standard Colab), you might need them.
# export OMP_NUM_THREADS=8 # Often managed by Colab/PyTorch automatically
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO # Can keep for debugging if needed, but remove for cleaner logs.

# --- Mamba Specific Configuration ---
# IMPORTANT: Update this to your Mamba model path or HF ID.
# In Colab, you'd typically download this to a local path or mount Google Drive.
# Example: If you download a model, it might be in /content/mamba-2.8b-hf
LLM_VERSION="state-spaces/mamba-2.8b" # <--- UPDATE THIS PATH
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEEN="${VISION_MODEL_VERSION//\//_}"

############### Finetune Stage ################

# Conversation template version. Start with 'vicuna_v1' or 'v0' for Mamba.
PROMPT_VERSION="vicuna_v1"

RUN_NAME="llava-mamba-${LLM_VERSION_CLEAN}-${VISION_MODEL_VERSION_CLEEN}-si_finetune_colab"
PREV_STAGE_CHECKPOINT="${LLM_VERSION}" # Assuming you start with the base Mamba model
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "RUN_NAME: ${RUN_NAME}"

# --- Training Command for Colab (Single-GPU A100) ---
# If you only have 1 GPU, use --nproc_per_node=1 and omit --nnodes, --node_rank, --master_addr, --master_port.
# If Colab provides multiple GPUs in a single instance (e.g., A100 x 2), set --nproc_per_node to that number.
# For typical A100 free/pro tier, it's often 1 GPU.
# Set NUM_GPUS based on what Colab assigns (usually 1). You might define it at the top or hardcode.
# Example: NUM_GPUS=1 if you have one GPU.
NUM_GPUS=1 # <--- Adjust based on your Colab instance's GPU count

ACCELERATE_CPU_AFFINITY=1 torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path "${PREV_STAGE_CHECKPOINT}" \
    --model_class_name "LlavaMamba" `# CRITICAL: Tells train.py to load your specific model wrapper`
    --version "${PROMPT_VERSION}" \
    # --- Colab Data Paths ---
    # You'll need to upload your data or mount Google Drive.
    # Example: /content/llava_data for images, /content/next_3p2m_single_image.yaml for data config
    --data_path "/content/next_3p2m_single_image.yaml" # <--- UPDATE THIS PATH
    --image_folder "/content/llava_data" # <--- UPDATE THIS PATH
    --video_folder "/content/llava_video" # <--- UPDATE THIS PATH (if you have video data)
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower "${VISION_MODEL_VERSION}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name "${RUN_NAME}" \
    # --- Colab Output Directory ---
    # Output checkpoints will go here. Consider mounting Google Drive for persistent storage.
    --output_dir "/content/checkpoints/${RUN_NAME}" # <--- UPDATE THIS PATH for outputs
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True `# Keep enabled, but disable if issues arise.`
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32

exit 0;