#!/bin/bash
# HybridPositionalEncoding3D 训练脚本 - 4卡 DeepSpeed ZeRO-2

cd /home/qyk/thinking-in-space

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 只使用4张卡
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Training HybridPositionalEncoding3D with DeepSpeed ZeRO-2 on 4 GPUs"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

deepspeed --num_gpus=4 train_3d_lora.py \
    --data_root /mnt/data/qyk/nyu-visionx/VSI-Bench \
    --video_root /mnt/data/qyk/43d_16f \
    --cache_dir /mnt/data/qyk/43d3dpt_16f \
    --output_dir ./checkpoints/3d_crossattn_16f_4gpu \
    --max_frames 16 \
    --batch_size 1 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --stage 1 \
    --train_projector \
    --deepspeed ds_config.json \
    --num_workers 4
