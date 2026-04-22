#!/bin/bash
# 残差 3D 注入 + DeepSpeed ZeRO-2 训练脚本

cd /home/qyk/thinking-in-space

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# DeepSpeed 环境变量
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1

# 获取可用 GPU 数量
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=1
fi
echo "Using $NUM_GPUS GPUs with DeepSpeed ZeRO-2"

deepspeed --num_gpus=$NUM_GPUS train_3d_residual.py \
    --data_root /mnt/data/qyk/nyu-visionx/VSI-Bench \
    --video_root /mnt/data/qyk/43d \
    --cache_dir /mnt/data/qyk/43d3dpt \
    --output_dir ./checkpoints/3d_residual_ds \
    --max_frames 32 \
    --batch_size 1 \
    --num_epochs 3 \
    --learning_rate 1e-3 \
    --warmup_ratio 0.1 \
    --train_projector \
    --hidden_dim 256 \
    --save_steps 500 \
    --num_workers 0 \
    --deepspeed ds_config_residual.json
