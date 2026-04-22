#!/bin/bash

# LLaVA-OneVision-3D 评测脚本
# 使用方法: bash evaluate_3d.sh --num_processes 8

set -e

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    gpu_count=${#devices[@]}
fi

benchmark=vsibench
output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")_3d
num_processes=4
num_frames=32
launcher=accelerate

# 3D 数据缓存路径
POINT_CLOUD_PATH="/mnt/data/qyk/43d3dpt"

# 3D Encoder 训练权重路径（可选）
CHECKPOINT_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
    --num_processes)
        num_processes="$2"
        shift 2
        ;;
    --output_path)
        output_path="$2"
        shift 2
        ;;
    --limit)
        limit="$2"
        shift 2
        ;;
    --num_frames)
        num_frames="$2"
        shift 2
        ;;
    --checkpoint)
        CHECKPOINT_PATH="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

echo "=============================================="
echo "LLaVA-OneVision-3D Evaluation"
echo "=============================================="
echo "Benchmark: $benchmark"
echo "Num Frames: $num_frames"
echo "Num Processes: $num_processes"
echo "Output Path: $output_path"
echo "Point Cloud Path: $POINT_CLOUD_PATH"
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint Path: $CHECKPOINT_PATH"
fi
echo "=============================================="

# 模型配置 - 3D 版本
model_family="llava_onevision_3d"
model="llava_one_vision_qwen2_0p5b_ov_${num_frames}f_3d"

# 模型参数 - 包含 3D 相关参数
model_args="pretrained=/mnt/data/qyk/lmms-lab/llava-onevision-qwen2-0.5b-ov/,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=$num_frames,use_3d=True,point_cloud_path=$POINT_CLOUD_PATH"

# 如果指定了 checkpoint，添加到模型参数
if [ -n "$CHECKPOINT_PATH" ]; then
    model_args="$model_args,checkpoint_path=$CHECKPOINT_PATH"
fi

# 构建启动命令
if [ "$launcher" = "python" ]; then
    export LMMS_EVAL_LAUNCHER="python"
    evaluate_script="python \
        "
elif [ "$launcher" = "accelerate" ]; then
    export LMMS_EVAL_LAUNCHER="accelerate"
    evaluate_script="accelerate launch \
        --num_processes=$num_processes \
        "
fi

evaluate_script="$evaluate_script -m lmms_eval \
    --model $model_family \
    --model_args $model_args \
    --tasks $benchmark \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $model \
    --output_path $output_path/$benchmark \
    "

if [ -n "$limit" ]; then
    evaluate_script="$evaluate_script \
        --limit $limit \
    "
fi

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
