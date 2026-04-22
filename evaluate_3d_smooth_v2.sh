#!/bin/bash

# LLaVA-OneVision-3D-Smooth-v2 评测脚本
# 使用方法: bash evaluate_3d_smooth_v2.sh --num_processes 4 --alpha 0.4

set -e

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    gpu_count=${#devices[@]}
fi

benchmark=vsibench
output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")_3d_smooth_v2
num_processes=4
num_frames=16
launcher=accelerate
alpha=0.4

# Matching groups 路径
MATCHING_GROUPS_PATH="/home/qyk/thinking-in-space/validation/43d3dpt_16f_match_voxel30cm"

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
    --alpha)
        alpha="$2"
        shift 2
        ;;
    --num_frames)
        num_frames="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

echo "========================================"
echo "Start evaluating 3D Smooth v2"
echo "========================================"
echo "Model: llava_onevision_3d_smooth_v2"
echo "Alpha: $alpha"
echo "Frames: $num_frames"
echo "Processes: $num_processes"
echo "Matching groups: $MATCHING_GROUPS_PATH"
echo "========================================"

ACCELERATE_CMD="/home/qyk/.conda/envs/vsibench/bin/accelerate"
PYTHON_CMD="/home/qyk/.conda/envs/vsibench/bin/python"

if [ "$launcher" = "python" ]; then
    export LMMS_EVAL_LAUNCHER="python"
    evaluate_script="$PYTHON_CMD \
        "
elif [ "$launcher" = "accelerate" ]; then
    export LMMS_EVAL_LAUNCHER="accelerate"
    evaluate_script="$ACCELERATE_CMD launch \
        --num_processes=$num_processes \
        "
fi

evaluate_script="$evaluate_script -m lmms_eval \
    --model llava_onevision_3d_smooth_v2 \
    --model_args pretrained=/mnt/data/qyk/lmms-lab/llava-onevision-qwen2-0.5b-ov/,matching_groups_path=$MATCHING_GROUPS_PATH,alpha=$alpha,max_frames_num=$num_frames \
    --tasks $benchmark \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix 3d_smooth_v2_${num_frames}f_alpha${alpha} \
    --output_path $output_path/$benchmark \
    "

echo $evaluate_script
eval $evaluate_script

echo "========================================"
echo "Evaluation complete!"
echo "Output: $output_path/$benchmark"
echo "========================================"
