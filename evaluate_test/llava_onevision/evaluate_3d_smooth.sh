#!/bin/bash
# LLaVA-OneVision with 3D Soft-Smoothing (通用版)

set -e

# 默认参数
NUM_PROCESSES=4
ALPHA=0.3
MODEL_NAME="llava-onevision-qwen2-0.5b-ov"  # 默认 0.5B
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
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--num_processes N] [--alpha 0.3] [--model MODEL_NAME] [--limit N]"
      echo "Example models: llava-onevision-qwen2-0.5b-ov, llava-onevision-qwen2-7b-ov"
      exit 1
      ;;
  esac
done

# 自动判断进程数（如果没指定）
if [[ "$MODEL_NAME" == *"7b"* ]] && [[ $NUM_PROCESSES == 4 ]]; then
  NUM_PROCESSES=8
fi

echo "=================================================="
echo "VSI-Bench: LLaVA-OneVision + 3D Smooth"
echo "=================================================="
echo "Model: $MODEL_NAME"
echo "Alpha: $ALPHA"
echo "Matching: $(basename $MATCHING_GROUPS_PATH)"
echo "Frames: 16"
echo "Processes: $NUM_PROCESSES"
echo "=================================================="

cd /home/qyk/thinking-in-space

export PYTHONPATH="/home/qyk/thinking-in-space:$PYTHONPATH"
export PYTHONPATH="/home/qyk/thinking-in-space/LLaVA-NeXT:$PYTHONPATH"
export LMMS_EVAL_LAUNCHER="accelerate"

MODEL_ARGS="pretrained=/mnt/data/qyk/lmms-lab/$MODEL_NAME/,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=16,mm_spatial_pool_stride=2,matching_groups_path=$MATCHING_GROUPS_PATH,alpha=$ALPHA"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
  LIMIT_ARG="--limit $LIMIT"
fi

OUTPUT_NAME=$(echo $MODEL_NAME | tr '-' '_')

accelerate launch \
    --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --model llava_onevision_3d_smooth \
    --model_args "$MODEL_ARGS" \
    --tasks vsibench \
    --batch_size 1 \
    $LIMIT_ARG \
    --log_samples \
    --log_samples_suffix ${OUTPUT_NAME}_smooth_${ALPHA} \
    --output_path ./logs/vsibench_${OUTPUT_NAME}_smooth_alpha${ALPHA}

echo "=================================================="
echo "Complete! Results: ./logs/vsibench_${OUTPUT_NAME}_smooth_alpha${ALPHA}"
echo "=================================================="
