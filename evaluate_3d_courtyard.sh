#!/bin/bash

# LLaVA-OneVision-3D Courtyard 专用评测脚本
# 仅测试 courtyard 视频，用于快速验证 3D 编码效果
# 使用方法: bash evaluate_3d_courtyard.sh

set -e

cd /home/qyk/thinking-in-space

benchmark=vsibench
output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")_3d_courtyard
num_frames=8  # courtyard 只有 38 帧，用 8 帧即可

# 3D 缓存目录（会自动指向 courtyard_3d.pt）
POINT_CLOUD_PATH="/home/qyk/map-anything/3d_cache"

echo "=============================================="
echo "LLaVA-OneVision-3D Courtyard Evaluation"
echo "=============================================="
echo "Benchmark: $benchmark"
echo "Num Frames: $num_frames"
echo "Output Path: $output_path"
echo "Point Cloud Path: $POINT_CLOUD_PATH"
echo "=============================================="
echo ""
echo "Finding courtyard samples in VSI-Bench..."

# 先查找有多少个 courtyard 样本
python -c "
import json
import sys

# 尝试找到 vsibench 的 annotation 文件
import os
for root, dirs, files in os.walk('/mnt/data/qyk/nyu-visionx/VSI-Bench'):
    for f in files:
        if f.endswith('.json') and 'anno' in f.lower():
            print(f'Found: {os.path.join(root, f)}')
" 2>/dev/null || true

# 设置 launcher 环境变量
export LMMS_EVAL_LAUNCHER="python"

# 模型配置 - 使用 courtyard 专用版本
model_family="llava_onevision_3d_courtyard"
model="llava_one_vision_qwen2_0p5b_ov_${num_frames}f_3d_courtyard"

# 模型参数
model_args="pretrained=/mnt/data/qyk/lmms-lab/llava-onevision-qwen2-0.5b-ov/,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=$num_frames,use_3d=True,point_cloud_path=$POINT_CLOUD_PATH"

evaluate_script="python -m lmms_eval \
    --model $model_family \
    --model_args $model_args \
    --tasks $benchmark \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $model \
    --output_path $output_path/$benchmark \
    "

echo ""
echo "Running command:"
echo "$evaluate_script"
echo ""
eval $evaluate_script

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: $output_path/$benchmark"
echo "=============================================="
