#!/bin/bash
# ============================================================================
# 轻量级路径适配脚本
# 用法: git clone --recursive https://github.com/Chocyork/thinking-in-space.git
#       cd thinking-in-space && bash setup_runpod.sh
# ============================================================================

set -e

# 配置（根据实际目录自动推断）
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(dirname "$PROJECT_DIR")"

echo "============================================================"
echo "  路径适配 + SigLIP Patch"
echo "  项目目录: ${PROJECT_DIR}"
echo "============================================================"

# ============================================================================
# 1. SigLIP Patch（LLaVA-NeXT builder.py）
# ============================================================================
echo "[1/2] 应用 SigLIP 优先检测补丁..."
BUILDER_PY="${PROJECT_DIR}/LLaVA-NeXT/llava/model/multimodal_encoder/builder.py"

if [ -f "${BUILDER_PY}" ]; then
    python3 << EOF
import re

path = "${BUILDER_PY}"
with open(path, 'r') as f:
    content = f.read()

if 'Fix: prioritize SigLIP' in content:
    print("  已 patch，跳过")
    exit(0)

old_pattern = r'(    vision_tower = getattr\(vision_tower_cfg, "mm_vision_tower", getattr\(vision_tower_cfg, "vision_tower", None\)\))\n'
replacement = r'''\1
    
    # Fix: prioritize SigLIP detection for local paths
    if "siglip" in vision_tower.lower():
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
'''

new_content = re.sub(old_pattern, replacement, content)

if new_content == content:
    print("  警告: 未找到匹配位置")
else:
    with open(path, 'w') as f:
        f.write(new_content)
    print("  SigLIP patch 应用成功")
EOF
else
    echo "  警告: 未找到 builder.py，跳过"
fi

# ============================================================================
# 2. 批量替换所有脚本中的旧路径
# ============================================================================
echo "[2/2] 替换脚本硬编码路径..."
cd "${PROJECT_DIR}"

# 推断常用路径（用于替换）
MODELS_DIR="${WORKSPACE}/models"
DATA_DIR="${WORKSPACE}/data"

ALL_SH=$(find . -maxdepth 2 -name "*.sh" -type f)

for f in ${ALL_SH}; do
    sed -i "s|/home/qyk/thinking-in-space|${PROJECT_DIR}|g" "$f"
    sed -i "s|/home/qyk/thinking-in-space/LLaVA-NeXT|${PROJECT_DIR}/LLaVA-NeXT|g" "$f"
    sed -i "s|/home/qyk/map-anything|${WORKSPACE}/map-anything|g" "$f"
    sed -i "s|/home/qyk/.conda/envs/vsibench/bin/|$(conda run -n base which python | xargs dirname)/|g" "$f" 2>/dev/null || true
    sed -i "s|/mnt/data/qyk/lmms-lab/|${MODELS_DIR}/|g" "$f"
    sed -i "s|/mnt/data/qyk/nyu-visionx/VSI-Bench|${DATA_DIR}/VSI-Bench|g" "$f"
    sed -i "s|/mnt/data/qyk/43d3dpt_16f|${DATA_DIR}/43d3dpt_16f|g" "$f"
    sed -i "s|/mnt/data/qyk/43d3dpt|${DATA_DIR}/43d3dpt|g" "$f"
    sed -i "s|/mnt/data/qyk/43d_16f|${DATA_DIR}/43d_16f|g" "$f"
    sed -i "s|/mnt/data/qyk/43d |${DATA_DIR}/43d |g" "$f"
    sed -i "s|/home/qyk/thinking-in-space/validation/|${PROJECT_DIR}/validation/|g" "$f"
done

# 修改 vsibench.yaml 的数据集路径
sed -i "s|dataset_path:.*|dataset_path: ${DATA_DIR}/VSI-Bench/|" lmms_eval/tasks/vsibench/vsibench.yaml 2>/dev/null || true

echo "============================================================"
echo "  完成"
echo "============================================================"
