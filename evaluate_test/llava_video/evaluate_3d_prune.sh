#!/bin/bash
# LLaVA-Video 7B with 3D Soft-Smoothing + Pruning (16 frames)

set -e

NUM_PROCESSES=${1:-8}
ALPHA=${2:-0.3}
MATCHING_GROUPS_PATH="/home/qyk/thinking-in-space/validation/43d3dpt_16f_match_voxel30cm"

echo "=================================================="
echo "VSI-Bench: LLaVA-Video-7B + 3D Prune"
echo "=================================================="
echo "Model: /mnt/data/qyk/lmms-lab/LLaVA-Video-7B-Qwen2/"
echo "Alpha: $ALPHA"
echo "Frames: 16"
echo "Processes: $NUM_PROCESSES"
echo "=================================================="

cd /home/qyk/thinking-in-space

export PYTHONPATH="/home/qyk/thinking-in-space:$PYTHONPATH"
export PYTHONPATH="/home/qyk/thinking-in-space/LLaVA-NeXT:$PYTHONPATH"
export LMMS_EVAL_LAUNCHER="accelerate"

MODEL_ARGS="pretrained=/mnt/data/qyk/lmms-lab/LLaVA-Video-7B-Qwen2/,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=16,mm_spatial_pool_stride=2,matching_groups_path=$MATCHING_GROUPS_PATH,alpha=$ALPHA"

accelerate launch \
    --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --model llava_onevision_3d_prune \
    --model_args "$MODEL_ARGS" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_video_7b_prune_${ALPHA} \
    --output_path ./logs/vsibench_llava_video_7b_prune_alpha${ALPHA}

echo "=================================================="
echo "Complete! Results: ./logs/vsibench_llava_video_7b_prune_alpha${ALPHA}"
echo "=================================================="
