#!/bin/bash
# 测试 llava_onevision_3d 评估器

cd /home/qyk/thinking-in-space

echo "=============================================="
echo "Testing LLaVA-OneVision-3D"
echo "=============================================="

# 设置环境变量
export LMMS_EVAL_LAUNCHER="python"

# 测试 3D 增强版本
echo ""
echo "[1] Testing 3D mode..."
python -m lmms_eval \
    --model llava_onevision_3d \
    --model_args "pretrained=/mnt/data/qyk/lmms-lab/llava-onevision-qwen2-0.5b-ov,use_3d=True,point_cloud_path=/home/qyk/map-anything/3d_cache/courtyard_3d.pt,max_frames_num=8" \
    --tasks vsibench \
    --batch_size 1 \
    --limit 1 \
    --log_samples \
    --log_samples_suffix test_3d \
    --output_path logs/test_3d 2>&1 | tee logs/test_3d.log

echo ""
echo "[2] Testing 2D mode (baseline)..."
python -m lmms_eval \
    --model llava_onevision_3d \
    --model_args "pretrained=/mnt/data/qyk/lmms-lab/llava-onevision-qwen2-0.5b-ov,use_3d=False,max_frames_num=8" \
    --tasks vsibench \
    --batch_size 1 \
    --limit 1 \
    --log_samples \
    --log_samples_suffix test_2d \
    --output_path logs/test_2d 2>&1 | tee logs/test_2d.log

echo ""
echo "=============================================="
echo "Test complete! Check logs/ for results"
echo "=============================================="
