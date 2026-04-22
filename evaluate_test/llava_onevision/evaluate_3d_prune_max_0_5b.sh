#!/bin/bash
# LLaVA-OneVision 0.5B with 3D Salience-Based Pruning (Max Norm) Evaluation

set -e

# 默认参数
NUM_PROCESSES=8
ALPHA=0.4  # 保留但不再使用（兼容性）
#MATCHING_GROUPS_PATH="/home/qyk/thinking-in-space/validation/43d3dpt_16f_match_voxel30cm"
#MATCHING_GROUPS_PATH="/home/qyk/thinking-in-space/validation/43d3dpt_16f_match_voxel10cm_iou45"
MATCHING_GROUPS_PATH="/home/qyk/thinking-in-space/validation/da3_2dir"

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
    --matching_groups)
      MATCHING_GROUPS_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "=================================================="
echo "VSI-Bench Evaluation: 0.5B + 3D Salience Pruning"
echo "=================================================="
echo "Model: llava-onevision-qwen2-0.5b-ov"
echo "Strategy: Keep token with MAX L2 norm in each group"
echo "Matching Groups: $MATCHING_GROUPS_PATH"
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
    --model llava_onevision_3d_prune_max \
    --model_args "$MODEL_ARGS" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision_0.5b_prune_max \
    --output_path ./logs/vsibench_0.5b_prune_max

echo "=================================================="
echo "Evaluation Complete!"
echo "Results: ./logs/vsibench_0.5b_prune_max"
echo "=================================================="
