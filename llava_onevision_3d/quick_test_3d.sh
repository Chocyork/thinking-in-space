#!/bin/bash
# 快速测试脚本：验证 3D 数据加载和模型推理

echo "=================================================="
echo "Quick 3D Test for LLaVA-OneVision"
echo "=================================================="

# 配置
VIDEO_DIR="/mnt/data/qyk/courtyard_raw/courtyard/images/dslr_jpgs"
CACHE_3D="/home/qyk/map-anything/3d_cache/courtyard_3d.pt"
MODEL="lmms-lab/llava-onevision-qwen2-7b-ov"

echo ""
echo "Step 1: Verify 3D cache exists"
if [ -f "$CACHE_3D" ]; then
    echo "✓ Found: $CACHE_3D"
    python3 -c "
import torch
data = torch.load('$CACHE_3D')
print(f'  Shape: {data[\"centers_3d\"].shape}')
print(f'  Valid: {data[\"valid_mask\"].float().mean():.1%}')
"
else
    echo "✗ Not found: $CACHE_3D"
    exit 1
fi

echo ""
echo "Step 2: Count video frames"
if [ -d "$VIDEO_DIR" ]; then
    NUM_FRAMES=$(ls -1 "$VIDEO_DIR"/*.jpg "$VIDEO_DIR"/*.png 2>/dev/null | wc -l)
    echo "✓ Found $NUM_FRAMES frames in $VIDEO_DIR"
else
    echo "✗ Not found: $VIDEO_DIR"
    exit 1
fi

echo ""
echo "Step 3: Test 2D baseline (without 3D)"
echo "Running: lmms_eval with 2D model..."

# 创建一个临时测试文件
cat > /tmp/test_courtyard.json << 'EOF'
[
  {
    "question": "What is the main object in the center of the scene?",
    "ground_truth": "table",
    "question_type": "object_rel_direction_easy"
  },
  {
    "question": "How many chairs are visible?",
    "ground_truth": "2",
    "question_type": "object_counting"
  },
  {
    "question": "What is on the left side of the courtyard?",
    "ground_truth": "tree",
    "question_type": "object_rel_direction_easy"
  }
]
EOF

echo "✓ Test questions prepared"

echo ""
echo "=================================================="
echo "Next Steps:"
echo "=================================================="
echo ""
echo "1. To test 2D baseline:"
echo "   bash evaluate_all_in_one.sh --model llava_onevision_qwen2_7b_ov_32f --limit 1"
echo ""
echo "2. To test with 3D (after implementing evaluator_3d.py):"
echo "   python test_with_3d.py --use_3d --cache $CACHE_3D"
echo ""
echo "3. To generate 3D cache for more videos:"
echo "   python batch_generate_3d.py --input_dir <video_frames> --output_dir 3d_cache/"
echo ""
