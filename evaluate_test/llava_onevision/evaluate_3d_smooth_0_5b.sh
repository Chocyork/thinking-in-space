#!/bin/bash
# LLaVA-OneVision with 3D Soft-Smoothing Feature Fusion Evaluation

set -e

# 默认参数
NUM_PROCESSES=4
BATCH_SIZE=1
ALPHA=0.3
MATCHING_GROUPS_PATH="/home/qyk/thinking-in-space/validation/43d3dpt_16f_match_voxel10cm_iou45"
NUM_FRAMES=16
LIMIT=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --num_processes)
      NUM_PROCESSES="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --alpha)
      ALPHA="$2"
      shift 2
      ;;
    --num_frames)
      NUM_FRAMES="$2"
      shift 2
      ;;
    --matching_groups_path)
      MATCHING_GROUPS_PATH="$2"
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
echo "VSI-Bench Evaluation with 3D Soft-Smoothing"
echo "=================================================="
echo "Model: llava_onevision_3d_smooth"
echo "Alpha: $ALPHA"
echo "Num Frames: $NUM_FRAMES"
echo "Matching Groups: $MATCHING_GROUPS_PATH"
echo "Num Processes: $NUM_PROCESSES"
if [ -n "$LIMIT" ]; then
  echo "Limit: $LIMIT samples"
fi
echo "=================================================="

# 切换到工作目录
cd /home/qyk/thinking-in-space

# 设置 Python 路径
export PYTHONPATH="/home/qyk/thinking-in-space:$PYTHONPATH"
export PYTHONPATH="/home/qyk/thinking-in-space/LLaVA-NeXT:$PYTHONPATH"

# 设置 launcher
export LMMS_EVAL_LAUNCHER="accelerate"

# 构建模型参数
MODEL_ARGS="pretrained=/mnt/data/qyk/lmms-lab/llava-onevision-qwen2-0.5b-ov/,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=$NUM_FRAMES,mm_spatial_pool_stride=2,matching_groups_path=$MATCHING_GROUPS_PATH,alpha=$ALPHA"

# 构建命令
EVAL_CMD="accelerate launch \
    --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --model llava_onevision_3d_smooth \
    --model_args "$MODEL_ARGS" \
    --tasks vsibench \
    --batch_size $BATCH_SIZE \
    --log_samples \
    --log_samples_suffix llava_onevision_3d_smooth_${ALPHA} \
    --output_path ./logs/vsibench_3d_smooth_alpha${ALPHA}"

# 添加 limit 参数
if [ -n "$LIMIT" ]; then
  EVAL_CMD="$EVAL_CMD --limit $LIMIT"
fi

echo "Running: $EVAL_CMD"
echo "=================================================="

# 运行评估
eval $EVAL_CMD

echo "=================================================="
echo "Evaluation Complete!"
echo "Results saved to: ./logs/vsibench_3d_smooth_alpha${ALPHA}"
echo "=================================================="
