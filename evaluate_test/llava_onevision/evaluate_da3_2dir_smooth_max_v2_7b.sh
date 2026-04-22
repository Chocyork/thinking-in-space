#!/bin/bash
# LLaVA-OneVision 7B with DA3-2dir + Smoothing + Max Pruning (v2)

set -e

NUM_PROCESSES=8
ALPHA=0.4
#MATCHING_GROUPS_PATH="/home/qyk/thinking-in-space/validation/da3_2dir"
MATCHING_GROUPS_PATH="/home/qyk/thinking-in-space/validation/43d3dpt_16f_match_voxel30cm"
#MATCHING_GROUPS_PATH="/home/qyk/thinking-in-space/validation/43d3dpt_16f_match_voxel10cm_iou45"
LIMIT=""

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
echo "VSI-Bench: 7B + DA3-2dir + Smooth+MaxPrune v2"
echo "=================================================="
echo "Model: llava-onevision-qwen2-7b-ov"
echo "Matching: DA3-2dir"
echo "Strategy: L2 Max -> Smoothing -> Pruning"
echo "Alpha: $ALPHA"
echo "Num Processes: $NUM_PROCESSES"
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
    --model llava_onevision_3d_smooth_max_v2 \
    --model_args "$MODEL_ARGS" \
    --tasks vsibench \
    --batch_size 1 \
    $LIMIT_ARG \
    --log_samples \
    --log_samples_suffix llava_7b_da3_smooth_max_v2_${ALPHA} \
    --output_path ./logs/vsibench_da3_smooth_max_v2_7b_${ALPHA}

echo "=================================================="
echo "Complete! Results: ./logs/vsibench_da3_smooth_max_v2_7b_${ALPHA}"
echo "=================================================="
