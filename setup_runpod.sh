#!/bin/bash
# ============================================================================
# RunPod 环境搭建脚本
# 目标: 所有内容严格放在 /workspace/yefei/ 下
# 用法: cd /workspace/yefei/vlm3d && bash setup_runpod.sh
# ============================================================================

set -e

# 配置 —— 严格限制在 /workspace/yefei/ 下
YEFEI_DIR="/workspace/yefei"
WORKSPACE="${YEFEI_DIR}/vlm3d"
PROJECT_DIR="${WORKSPACE}/thinking-in-space"
MODELS_DIR="${WORKSPACE}/models"
DATA_DIR="${WORKSPACE}/data"
CONDA_ROOT="${YEFEI_DIR}/miniconda3"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  VLM3D RunPod 环境搭建${NC}"
echo -e "${BLUE}  根目录: ${YEFEI_DIR}${NC}"
echo -e "${BLUE}  工作区: ${WORKSPACE}${NC}"
echo -e "${BLUE}============================================================${NC}"

# ============================================================================
# Step 1: 确认 conda 存在
# ============================================================================
if [ ! -d "${CONDA_ROOT}" ]; then
    echo -e "${YELLOW}错误: 未找到 ${CONDA_ROOT}${NC}"
    echo "请先安装 miniconda 到 /workspace/yefei/miniconda3"
    exit 1
fi

export PATH="${CONDA_ROOT}/bin:$PATH"
echo -e "${GREEN}[Step 1/8] Conda 已确认: ${CONDA_ROOT}${NC}"

# ============================================================================
# Step 2: Clone 代码仓库
# ============================================================================
echo -e "${YELLOW}[Step 2/8] Clone 代码仓库...${NC}"
mkdir -p ${WORKSPACE}

if [ ! -d "${PROJECT_DIR}" ]; then
    cd ${WORKSPACE}
    git clone --recursive https://github.com/Chocyork/thinking-in-space.git
    cd ${PROJECT_DIR}
    git submodule update --init --recursive
else
    echo -e "${GREEN}thinking-in-space 已存在，跳过 clone${NC}"
fi

echo -e "${GREEN}[Step 2/8] 代码仓库完成${NC}"

# ============================================================================
# Step 3: 创建 vsibench 环境
# ============================================================================
echo -e "${YELLOW}[Step 3/8] 创建 vsibench Python 环境...${NC}"
if ! conda env list | grep -q "vlm3d_vsibench"; then
    conda create --name vlm3d_vsibench python=3.10 -y
fi

echo -e "${YELLOW}[Step 3/8] 安装核心依赖...${NC}"
cd ${PROJECT_DIR}
conda run -n vlm3d_vsibench pip install --no-cache-dir -e .
conda run -n vlm3d_vsibench pip install --no-cache-dir -e ./transformers
conda run -n vlm3d_vsibench pip install --no-cache-dir s2wrapper@git+https://github.com/bfshi/scaling_on_scales
conda run -n vlm3d_vsibench pip install --no-cache-dir deepspeed
conda run -n vlm3d_vsibench pip install --no-cache-dir decord

echo -e "${GREEN}[Step 3/8] vsibench 环境完成${NC}"

# ============================================================================
# Step 4: 创建 mapanything 环境（3D 重建用）
# ============================================================================
echo -e "${YELLOW}[Step 4/8] 创建 mapanything 环境...${NC}"
if ! conda env list | grep -q "vlm3d_mapanything"; then
    conda create --name vlm3d_mapanything python=3.10 -y
fi

# 预装 CUDA 版 torch
echo "  安装 PyTorch (CUDA)..."
conda run -n vlm3d_mapanything pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Clone 并安装 Depth-Anything-3
if [ ! -d "${WORKSPACE}/Depth-Anything-3" ]; then
    echo "  Clone Depth-Anything-3..."
    cd ${WORKSPACE}
    git clone https://github.com/Chocyork/Depth-Anything-3.git
fi
cd ${WORKSPACE}/Depth-Anything-3
conda run -n vlm3d_mapanything pip install --no-cache-dir -e .

# Clone 并安装 MapAnything
if [ ! -d "${WORKSPACE}/map-anything" ]; then
    echo "  Clone map-anything..."
    cd ${WORKSPACE}
    git clone https://github.com/Chocyork/map-anything.git
fi
cd ${WORKSPACE}/map-anything
conda run -n vlm3d_mapanything pip install --no-cache-dir -e ".[colmap]"

echo -e "${GREEN}[Step 4/8] mapanything 环境完成${NC}"

# ============================================================================
# Step 5: 下载模型权重
# ============================================================================
echo -e "${YELLOW}[Step 5/8] 下载模型权重...${NC}"
mkdir -p ${MODELS_DIR}

# 0.5B
if [ ! -d "${MODELS_DIR}/llava-onevision-qwen2-0.5b-ov" ]; then
    echo "下载 llava-onevision-qwen2-0.5b-ov..."
    conda run -n vlm3d_vsibench huggingface-cli download \
        lmms-lab/llava-onevision-qwen2-0.5b-ov \
        --local-dir ${MODELS_DIR}/llava-onevision-qwen2-0.5b-ov
fi

# 7B
if [ ! -d "${MODELS_DIR}/llava-onevision-qwen2-7b-ov" ]; then
    echo "下载 llava-onevision-qwen2-7b-ov..."
    conda run -n vlm3d_vsibench huggingface-cli download \
        lmms-lab/llava-onevision-qwen2-7b-ov \
        --local-dir ${MODELS_DIR}/llava-onevision-qwen2-7b-ov
fi

# Video 7B
if [ ! -d "${MODELS_DIR}/LLaVA-Video-7B-Qwen2" ]; then
    echo "下载 LLaVA-Video-7B-Qwen2..."
    conda run -n vlm3d_vsibench huggingface-cli download \
        lmms-lab/LLaVA-Video-7B-Qwen2 \
        --local-dir ${MODELS_DIR}/LLaVA-Video-7B-Qwen2
fi

echo -e "${GREEN}[Step 5/8] 模型下载完成${NC}"

# ============================================================================
# Step 6: 下载 VSI-Bench 数据集
# ============================================================================
echo -e "${YELLOW}[Step 6/8] 下载 VSI-Bench 数据集...${NC}"
mkdir -p ${DATA_DIR}

conda run -n vlm3d_vsibench huggingface-cli download \
    nyu-visionx/VSI-Bench \
    --local-dir ${DATA_DIR}/VSI-Bench \
    --repo-type dataset

echo -e "${GREEN}[Step 6/8] 数据集下载完成${NC}"

# ============================================================================
# Step 7: 创建数据目录结构
# ============================================================================
echo -e "${YELLOW}[Step 7/8] 创建数据目录结构...${NC}"
mkdir -p ${DATA_DIR}/43d_16f
mkdir -p ${DATA_DIR}/43d3dpt_16f

echo -e "${GREEN}[Step 7/8] 目录结构完成${NC}"

# ============================================================================
# Step 8: 全面替换所有脚本中的硬编码路径
# ============================================================================
echo -e "${YELLOW}[Step 8/8] 替换所有脚本硬编码路径...${NC}"
cd ${PROJECT_DIR}

# 收集所有需要修改的 .sh 文件
ALL_SH=$(find . -maxdepth 2 -name "*.sh" -type f)

for f in ${ALL_SH}; do
    # 1. 代码目录
    sed -i "s|/home/qyk/thinking-in-space|${WORKSPACE}/thinking-in-space|g" "$f"
    # 2. LLaVA-NeXT PYTHONPATH
    sed -i "s|/home/qyk/thinking-in-space/LLaVA-NeXT|${WORKSPACE}/thinking-in-space/LLaVA-NeXT|g" "$f"
    # 3. map-anything 路径
    sed -i "s|/home/qyk/map-anything|${WORKSPACE}/map-anything|g" "$f"
    # 4. conda 环境路径 (accelerate/python)
    sed -i "s|/home/qyk/.conda/envs/vsibench/bin/|${CONDA_ROOT}/envs/vlm3d_vsibench/bin/|g" "$f"
    # 5. 模型权重路径
    sed -i "s|/mnt/data/qyk/lmms-lab/|${MODELS_DIR}/|g" "$f"
    # 6. 数据集路径 (先精确匹配长的，再匹配短的)
    sed -i "s|/mnt/data/qyk/nyu-visionx/VSI-Bench|${DATA_DIR}/VSI-Bench|g" "$f"
    # 7. 3D 缓存路径 (先替换带后缀的，避免前缀冲突)
    sed -i "s|/mnt/data/qyk/43d3dpt_16f|${DATA_DIR}/43d3dpt_16f|g" "$f"
    sed -i "s|/mnt/data/qyk/43d3dpt|${DATA_DIR}/43d3dpt|g" "$f"
    # 8. 帧图片路径
    sed -i "s|/mnt/data/qyk/43d_16f|${DATA_DIR}/43d_16f|g" "$f"
    sed -i "s|/mnt/data/qyk/43d |${DATA_DIR}/43d |g" "$f"
    # 9. validation 路径 (matching groups)
    sed -i "s|/home/qyk/thinking-in-space/validation/|${PROJECT_DIR}/validation/|g" "$f"
done

# 修改 vsibench.yaml 的数据集路径
sed -i \
    "s|dataset_path:.*|dataset_path: ${DATA_DIR}/VSI-Bench/|" \
    lmms_eval/tasks/vsibench/vsibench.yaml

echo -e "${GREEN}[Step 8/8] 路径修改完成${NC}"

# ============================================================================
# 完成
# ============================================================================
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  环境搭建完成！${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "激活环境:"
echo "  conda activate vlm3d_vsibench      # 用于评测"
echo "  conda activate vlm3d_mapanything   # 用于 3D 重建"
echo ""
echo "常用路径:"
echo "  代码:     ${PROJECT_DIR}"
echo "  模型:     ${MODELS_DIR}"
echo "  数据:     ${DATA_DIR}"
echo ""
echo "下一步:"
echo "  1. 把 matching_groups_core.tar.gz 上传到 ${PROJECT_DIR}/ 并解压"
echo "  2. 运行 extract_frames_for_3d.py 生成 16 帧帧图片"
echo "  3. 运行 step1_batch_processor.py 生成 3D 缓存"
echo "  4. 运行评测脚本"
echo ""
