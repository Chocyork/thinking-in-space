#!/bin/bash
# Debug version - 确认剪枝是否生效

set -e

NUM_PROCESSES=1  # 强制单进程，便于调试
ALPHA=0.4
MATCHING_GROUPS_PATH="/home/qyk/thinking-in-space/validation/43d3dpt_16f_match_voxel30cm"
LIMIT=5  # 只跑5个样本

echo "=================================================="
echo "DEBUG: 7B Pruning Verification"
echo "=================================================="
echo "Model: llava-onevision-qwen2-7b-ov"
echo "Matching: $MATCHING_GROUPS_PATH"
echo "Alpha: $ALPHA"
echo "Processes: $NUM_PROCESSES (单进程用于调试)"
echo "Limit: $LIMIT"
echo "=================================================="

# 检查 matching_groups 是否存在
echo ""
echo "[检查] Matching groups 目录:"
ls -la "$MATCHING_GROUPS_PATH" | head -10

# 统计有多少个视频有 matching_groups.json
echo ""
echo "[统计] 有效视频数:"
find "$MATCHING_GROUPS_PATH" -name "matching_groups.json" | wc -l

# 显示前3个视频的 matching_groups 大小
echo ""
echo "[示例] 前3个视频的匹配组大小:"
for f in $(find "$MATCHING_GROUPS_PATH" -name "matching_groups.json" | head -3); do
    size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo "unknown")
    echo "  $f: ${size} bytes"
done

echo ""
echo "=================================================="
echo "开始评测（观察输出中的 🔥 标记确认剪枝）"
echo "=================================================="

cd /home/qyk/thinking-in-space

export PYTHONPATH="/home/qyk/thinking-in-space:$PYTHONPATH"
export PYTHONPATH="/home/qyk/thinking-in-space/LLaVA-NeXT:$PYTHONPATH"
export LMMS_EVAL_LAUNCHER="accelerate"

MODEL_ARGS="pretrained=/mnt/data/qyk/lmms-lab/llava-onevision-qwen2-7b-ov/,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=16,mm_spatial_pool_stride=2,matching_groups_path=$MATCHING_GROUPS_PATH,alpha=$ALPHA"

accelerate launch \
    --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --model llava_onevision_3d_smooth_max_v2 \
    --model_args "$MODEL_ARGS" \
    --tasks vsibench \
    --batch_size 1 \
    --limit $LIMIT \
    --log_samples \
    --log_samples_suffix debug_pruning \
    --output_path ./logs/debug_pruning_7b 2>&1 | tee /tmp/pruning_debug.log

echo ""
echo "=================================================="
echo "检查日志中的剪枝标记:"
echo "=================================================="
grep -E "(🔥|\[Smooth\+|剪枝|prune|成功锁定)" /tmp/pruning_debug.log || echo "⚠️ 未找到剪枝标记！"

echo ""
echo "完整日志保存在: /tmp/pruning_debug.log"
echo "=================================================="
