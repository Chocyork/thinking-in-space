#!/bin/bash
# ============================================================================
# RunPod 一键环境搭建脚本
# 用法: bash setup_runpod.sh
# ============================================================================

set -e

# 配置
WORKSPACE="/workspace"
PROJECT_DIR="${WORKSPACE}/thinking-in-space"
MODELS_DIR="${WORKSPACE}/models"
DATA_DIR="${WORKSPACE}/data"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Thinking-in-Space RunPod 环境搭建${NC}"
echo -e "${BLUE}============================================================${NC}"

# ============================================================================
# Step 1: 安装 miniconda（如果还没有）
# ============================================================================
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}[Step 1/8] 安装 Miniconda...${NC}"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p ${WORKSPACE}/miniconda3
    rm /tmp/miniconda.sh
    export PATH="${WORKSPACE}/miniconda3/bin:$PATH"
    echo 'export PATH="/workspace/miniconda3/bin:$PATH"' >> ~/.bashrc
    conda init bash
else
    echo -e "${GREEN}[Step 1/8] Conda 已存在，跳过${NC}"
fi

# ============================================================================
# Step 2: Clone 代码仓库
# ============================================================================
echo -e "${YELLOW}[Step 2/8] Clone 代码仓库...${NC}"
if [ ! -d "${PROJECT_DIR}" ]; then
    cd ${WORKSPACE}
    git clone --recursive https://github.com/Chocyork/thinking-in-space.git
    cd ${PROJECT_DIR}
    # 更新子模块
    git submodule update --init --recursive
else
    echo -e "${GREEN}代码已存在，跳过 clone${NC}"
    cd ${PROJECT_DIR}
fi

# ============================================================================
# Step 3: 创建 vsibench 环境
# ============================================================================
echo -e "${YELLOW}[Step 3/8] 创建 vsibench Python 环境...${NC}"
if ! conda env list | grep -q "vsibench"; then
    conda create --name vsibench python=3.10 -y
fi

echo -e "${YELLOW}[Step 3/8] 安装核心依赖...${NC}"
conda run -n vsibench pip install --no-cache-dir -e .
conda run -n vsibench pip install --no-cache-dir -e ./transformers
conda run -n vsibench pip install --no-cache-dir s2wrapper@git+https://github.com/bfshi/scaling_on_scales
conda run -n vsibench pip install --no-cache-dir deepspeed
conda run -n vsibench pip install --no-cache-dir decord

echo -e "${GREEN}[Step 3/8] vsibench 环境完成${NC}"

# ============================================================================
# Step 4: 创建 mapanything 环境（3D 重建用）
# ============================================================================
echo -e "${YELLOW}[Step 4/8] 创建 mapanything 环境...${NC}"
if ! conda env list | grep -q "mapanything"; then
    conda create --name mapanything python=3.10 -y
fi

# MapAnything 的依赖需要手动安装，这里只创建环境
# 具体依赖请参考 /workspace/map-anything 下的 README
echo -e "${GREEN}[Step 4/8] mapanything 环境已创建（依赖需手动安装）${NC}"

# ============================================================================
# Step 5: 下载模型权重
# ============================================================================
echo -e "${YELLOW}[Step 5/8] 下载模型权重...${NC}"
mkdir -p ${MODELS_DIR}

# 0.5B
if [ ! -d "${MODELS_DIR}/llava-onevision-qwen2-0.5b-ov" ]; then
    echo "下载 llava-onevision-qwen2-0.5b-ov..."
    conda run -n vsibench huggingface-cli download \
        lmms-lab/llava-onevision-qwen2-0.5b-ov \
        --local-dir ${MODELS_DIR}/llava-onevision-qwen2-0.5b-ov
fi

# 7B
if [ ! -d "${MODELS_DIR}/llava-onevision-qwen2-7b-ov" ]; then
    echo "下载 llava-onevision-qwen2-7b-ov..."
    conda run -n vsibench huggingface-cli download \
        lmms-lab/llava-onevision-qwen2-7b-ov \
        --local-dir ${MODELS_DIR}/llava-onevision-qwen2-7b-ov
fi

# Video 7B
if [ ! -d "${MODELS_DIR}/LLaVA-Video-7B-Qwen2" ]; then
    echo "下载 LLaVA-Video-7B-Qwen2..."
    conda run -n vsibench huggingface-cli download \
        lmms-lab/LLaVA-Video-7B-Qwen2 \
        --local-dir ${MODELS_DIR}/LLaVA-Video-7B-Qwen2
fi

echo -e "${GREEN}[Step 5/8] 模型下载完成${NC}"

# ============================================================================
# Step 6: 下载 VSI-Bench 数据集
# ============================================================================
echo -e "${YELLOW}[Step 6/8] 下载 VSI-Bench 数据集...${NC}"
mkdir -p ${DATA_DIR}

conda run -n vsibench huggingface-cli download \
    nyu-visionx/VSI-Bench \
    --local-dir ${DATA_DIR}/VSI-Bench \
    --repo-type dataset

echo -e "${GREEN}[Step 6/8] 数据集下载完成${NC}"

# ============================================================================
# Step 7: 创建数据目录软链接
# ============================================================================
echo -e "${YELLOW}[Step 7/8] 创建目录结构...${NC}"
mkdir -p ${DATA_DIR}/43d_16f
mkdir -p ${DATA_DIR}/43d3dpt_16f

# 创建常用路径的软链接（兼容旧脚本中的 hardcoded 路径）
mkdir -p /mnt/data/qyk
ln -sf ${DATA_DIR} /mnt/data/qyk/data
ln -sf ${MODELS_DIR} /mnt/data/qyk/models
ln -sf ${DATA_DIR}/43d_16f /mnt/data/qyk/43d_16f
ln -sf ${DATA_DIR}/43d3dpt_16f /mnt/data/qyk/43d3dpt_16f
ln -sf ${DATA_DIR}/VSI-Bench /mnt/data/qyk/nyu-visionx/VSI-Bench

echo -e "${GREEN}[Step 7/8] 目录结构完成${NC}"

# ============================================================================
# Step 8: 修改脚本路径（可选）
# ============================================================================
echo -e "${YELLOW}[Step 8/8] 修改 evaluate 脚本路径...${NC}"
cd ${PROJECT_DIR}

# 把所有 evaluate 脚本中的 /mnt/data/qyk/lmms-lab/ 替换为 /workspace/models/
find evaluate_test -name "*.sh" -exec sed -i \
    's|/mnt/data/qyk/lmms-lab/|/workspace/models/|g' {} \;

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
echo "  conda activate vsibench    # 用于评测"
echo "  conda activate mapanything # 用于 3D 重建"
echo ""
echo "常用路径:"
echo "  代码:     ${PROJECT_DIR}"
echo "  模型:     ${MODELS_DIR}"
echo "  数据:     ${DATA_DIR}"
echo ""
echo "下一步:"
echo "  1. 把 matching_groups_core.tar.gz 上传到 ${PROJECT_DIR}/ 并解压"
echo "  2. 运行 extract_frames_for_3d.py 生成 16 帧帧图片"
echo "  3. 运行 MapAnything 3D 重建"
echo "  4. 运行评测脚本"
echo ""
