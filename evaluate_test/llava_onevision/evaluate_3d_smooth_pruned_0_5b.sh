#!/bin/bash
# LLaVA-OneVision with 3D Soft-Smoothing PRUNED Evaluation

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
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "=================================================="
echo "VSI-Bench Evaluation with 3D Prune (Simplified)"
echo "=================================================="
echo "Model: llava_onevision_3d_prune (0.5B)"
echo "Alpha: $ALPHA"
echo "Strategy: Pruning (只保留每组第一个token)"
echo "Num Processes: $NUM_PROCESSES"
echo "=================================================="

cd /home/qyk/thinking-in-space

export PYTHONPATH="/home/qyk/thinking-in-space:$PYTHONPATH"
export PYTHONPATH="/home/qyk/thinking-in-space/LLaVA-NeXT:$PYTHONPATH"
export LMMS_EVAL_LAUNCHER="accelerate"

MODEL_ARGS="pretrained=/mnt/data/qyk/lmms-lab/llava-onevision-qwen2-0.5b-ov/,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=16,mm_spatial_pool_stride=2,matching_groups_path=$MATCHING_GROUPS_PATH,alpha=$ALPHA"

accelerate launch \
    --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --model llava_onevision_3d_prune \
    --model_args "$MODEL_ARGS" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision_3d_prune_${ALPHA} \
    --output_path ./logs/vsibench_3d_prune_alpha${ALPHA}

echo "=================================================="
echo "Evaluation Complete!"
echo "Results: ./logs/vsibench_3d_prune_alpha${ALPHA}}"
echo "=================================================="
