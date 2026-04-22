#!/bin/bash

set -e

# 自动检测 GPU 数量
if command -v nvidia-smi &> /dev/null; then
    auto_gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    auto_gpu_count=1
fi

# 参数解析
num_processes=$auto_gpu_count
dataset_path="/mnt/data/qyk/nyu-visionx/VSI-Bench/"

while [[ $# -gt 0 ]]; do
  case $1 in
    --num_processes)
      num_processes="$2"
      shift 2
      ;;
    --dataset_path)
      dataset_path="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

export OPENAI_API_KEY=""
export GOOGLE_API_KEY=""

cd /home/qyk/thinking-in-space

export PYTHONPATH="/home/qyk/thinking-in-space:$PYTHONPATH"
export PYTHONPATH="/home/qyk/thinking-in-space/LLaVA-NeXT:$PYTHONPATH"
export LMMS_EVAL_LAUNCHER="accelerate"

echo "=================================================="
echo "Baseline 7B 16f Evaluation"
echo "=================================================="
echo "Model: llava-onevision-qwen2-7b-ov"
echo "Frames: 16"
echo "Processes: $num_processes"
echo "Dataset: $dataset_path"
echo "=================================================="

MODEL_ARGS="pretrained=/mnt/data/qyk/lmms-lab/llava-onevision-qwen2-7b-ov/,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=16,mm_spatial_pool_stride=2"

# 使用环境变量传递数据集路径（任务配置会读取）
export VSIBENCH_DATASET_PATH="$dataset_path"

accelerate launch \
    --num_processes=$num_processes \
    -m lmms_eval \
    --model llava_onevision \
    --model_args "$MODEL_ARGS" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision_7b_16f_baseline \
    --output_path ./logs/vsibench_baseline_7b_16f

echo "=================================================="
echo "Complete! Results: ./logs/vsibench_baseline_7b_16f"
echo "=================================================="
