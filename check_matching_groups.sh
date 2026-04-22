#!/bin/bash
# 检查 matching_groups 中的视频ID格式

echo "=================================================="
echo "Matching Groups 检查工具"
echo "=================================================="

MATCHING_GROUPS_PATH="${1:-/home/qyk/thinking-in-space/validation/43d3dpt_16f_match_voxel30cm}"

echo ""
echo "检查目录: $MATCHING_GROUPS_PATH"
echo ""

# 显示前10个视频ID
echo "前10个视频的ID:"
ls "$MATCHING_GROUPS_PATH" | head -10 | while read dir; do
    if [ -f "$MATCHING_GROUPS_PATH/$dir/matching_groups.json" ]; then
        echo "  $dir"
    fi
done

echo ""
echo "总视频数: $(find "$MATCHING_GROUPS_PATH" -name 'matching_groups.json' | wc -l)"

echo ""
echo "示例视频ID格式:"
for dir in $(ls "$MATCHING_GROUPS_PATH" | head -3); do
    if [ -f "$MATCHING_GROUPS_PATH/$dir/matching_groups.json" ]; then
        echo ""
        echo "视频: $dir"
        python3 << EOF
import json
with open("$MATCHING_GROUPS_PATH/$dir/matching_groups.json") as f:
    data = json.load(f)
    print(f"  JSON中的video_id: {data.get('video_id', 'N/A')}")
    print(f"  组数: {len(data.get('groups', []))}")
    if data.get('groups'):
        g = data['groups'][0]
        print(f"  第一组大小: {len(g.get('members', []))}")
EOF
    fi
done

echo ""
echo "=================================================="
echo "提示: 确保评测时提取的视频ID与上述格式一致"
echo "=================================================="
