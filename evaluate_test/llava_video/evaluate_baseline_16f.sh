#!/bin/bash
# LLaVA-Video 7B Baseline Evaluation (16 frames)

set -e

NUM_PROCESSES=${1:-8}
DATASET_PATH=${2:-"/mnt/data/qyk/43d_16f/"}

echo "=================================================="
echo "VSI-Bench Baseline: LLaVA-Video-7B-Qwen2 (16f)"
echo "=================================================="
echo "Model: /mnt/data/qyk/lmms-lab/LLaVA-Video-7B-Qwen2/"
echo "Dataset: $DATASET_PATH"
echo "Frames: 16"
echo "Processes: $NUM_PROCESSES"
echo "=================================================="

cd /home/qyk/thinking-in-space

export PYTHONPATH="/home/qyk/thinking-in-space:$PYTHONPATH"
export PYTHONPATH="/home/qyk/thinking-in-space/LLaVA-NeXT:$PYTHONPATH"
export LMMS_EVAL_LAUNCHER="accelerate"
export VSIBENCH_DATASET_PATH="$DATASET_PATH"

MODEL_ARGS="pretrained=/mnt/data/qyk/lmms-lab/LLaVA-Video-7B-Qwen2/,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=16"

accelerate launch \
    --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --model llava_onevision \
    --model_args "$MODEL_ARGS" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_video_7b_16f_baseline \
    --output_path ./logs/vsibench_llava_video_7b_baseline_16f

echo "=================================================="
echo "Complete! Results: ./logs/vsibench_llava_video_7b_baseline_16f"
echo "=================================================="
