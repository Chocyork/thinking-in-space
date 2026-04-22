#!/bin/bash
# 诊断 Pruned 版为什么 7B 不生效

echo "=========================================="
echo "诊断: 检查 vid 和 keep_mask"
echo "=========================================="

# 先修改代码添加调试打印
sed -i 's/# print(f"\[Stage1\]/print(f"[DEBUG Stage1]/g' /home/qyk/thinking-in-space/lmms_eval/models/llava_onevision_3d_smooth_pruned.py
sed -i 's/_global_keep_mask = None/_global_keep_mask = None\n        print(f"[DEBUG] vid={video_id}, in_cache={video_id in _matching_groups if _matching_groups else False}")/g' /home/qyk/thinking-in-space/lmms_eval/models/llava_onevision_3d_smooth_pruned.py

echo ""
echo "=== 测试 0.5B (应该看到 keep_mask) ==="
CUDA_VISIBLE_DEVICES=0 bash evaluate_3d_smooth_pruned.sh --alpha 0.3 --limit 1 --num_processes 1 2>&1 | grep -E "(DEBUG|Pruned|keep)" | head -10

echo ""
echo "=== 测试 7B (检查差异) ==="
CUDA_VISIBLE_DEVICES=0 bash evaluate_3d_smooth_pruned_7b.sh --alpha 0.3 --limit 1 --num_processes 1 2>&1 | grep -E "(DEBUG|Pruned|keep)" | head -10

# 恢复代码
sed -i 's/print(f"\[DEBUG Stage1\]/# print(f"[Stage1]/g' /home/qyk/thinking-in-space/lmms_eval/models/llava_onevision_3d_smooth_pruned.py

echo ""
echo "=========================================="
echo "诊断完成"
echo "=========================================="
