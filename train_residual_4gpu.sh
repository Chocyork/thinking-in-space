#!/bin/bash
# 残差 3D 注入训练脚本 - 4卡版本

cd /home/qyk/thinking-in-space

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 使用4张卡
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Training with Residual 3D Injection on 4 GPUs (~100K params)"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_3d_residual.py \
    --data_root /mnt/data/qyk/nyu-visionx/VSI-Bench \
    --video_root /mnt/data/qyk/43d_16f \
    --cache_dir /mnt/data/qyk/43d3dpt_16f \
    --output_dir ./checkpoints/3d_residual_16f_4gpu_5 \
    --max_frames 16 \
    --batch_size 1 \
    --num_epochs 5 \
    --learning_rate 1e-3 \
    --warmup_ratio 0.1 \
    --train_projector \
    --hidden_dim 256 \
    --save_steps 500 \
    --num_workers 4
