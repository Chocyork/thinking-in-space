#!/bin/bash
# 残差 3D 注入训练脚本 - 轻量级，单卡友好

cd /home/qyk/thinking-in-space

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 选择 GPU（单卡即可，残差注入很轻量）
export CUDA_VISIBLE_DEVICES=0

echo "Training with Residual 3D Injection (~100K params)"
echo "This is much lighter than cross-attention (4.5M params)"

python train_3d_residual.py \
    --data_root /mnt/data/qyk/nyu-visionx/VSI-Bench \
    --video_root /mnt/data/qyk/43d \
    --cache_dir /mnt/data/qyk/43d3dpt \
    --output_dir ./checkpoints/3d_residual \
    --max_frames 32 \
    --batch_size 1 \
    --num_epochs 3 \
    --learning_rate 1e-3 \
    --warmup_ratio 0.1 \
    --train_projector \
    --hidden_dim 256 \
    --save_steps 500 \
    --num_workers 0
