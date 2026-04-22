#!/bin/bash
# LLaVA-OneVision 7B with 3D Soft-Smoothing Evaluation

set -e

# 默认参数
NUM_PROCESSES=8
ALPHA=0.3
MATCHING_GROUPS_PATH="/home/qyk/thinking-in-space/validation/43d3dpt_16f_match_voxel10cm_iou45"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --num_processes)
      NUM_PROCESSES="$2"
      shift 2
      ;;
    --alpha)
      ALPHA="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "=================================================="
echo "VSI-Bench Evaluation: 7B + 3D Smooth"
echo "=================================================="
echo "Model: llava-onevision-qwen2-7b-ov"
echo "Alpha: $ALPHA"
echo "Frames: 16"
echo "Processes: $NUM_PROCESSES"
echo "=================================================="

cd /home/qyk/thinking-in-space

export PYTHONPATH="/home/qyk/thinking-in-space:$PYTHONPATH"
export PYTHONPATH="/home/qyk/thinking-in-space/LLaVA-NeXT:$PYTHONPATH"
export LMMS_EVAL_LAUNCHER="accelerate"

MODEL_ARGS="pretrained=/mnt/data/qyk/lmms-lab/llava-onevision-qwen2-7b-ov/,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=16,mm_spatial_pool_stride=2,matching_groups_path=$MATCHING_GROUPS_PATH,alpha=$ALPHA"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
  LIMIT_ARG="--limit $LIMIT"
fi

accelerate launch \
    --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --model llava_onevision_3d_smooth \
    --model_args "$MODEL_ARGS" \
    --tasks vsibench \
    --batch_size 1 \
    $LIMIT_ARG \
    --log_samples \
    --log_samples_suffix llava_onevision_7b_smooth_${ALPHA} \
    --output_path ./logs/vsibench_7b_smooth_alpha${ALPHA}

echo "=================================================="
echo "Complete! Results: ./logs/vsibench_7b_smooth_alpha${ALPHA}"
echo "=================================================="
