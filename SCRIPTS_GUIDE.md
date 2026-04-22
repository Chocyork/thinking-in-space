# 项目文件使用说明

本文档列出 `thinking-in-space` 项目下所有 Python 和 Shell 脚本的功能及使用方法。

---

## 📦 Python 脚本


#### `train_3d_residual.py`
- **功能**：Residual3DInjection 模型（轻量级，366K参数）
- **特点**：
  - 无 Cross-Attention，显存友好
  - 支持单卡和 DDP 多卡训练
  - 适合快速实验
- **使用**：
  ```bash
  # 单卡
  python train_3d_residual.py --max_frames 16 --batch_size 1
  
  # 多卡（使用torchrun）
  torchrun --nproc_per_node=4 train_3d_residual.py --max_frames 16
  ```

#### `train_3d_lora.py`（未确定，不一定用）
- **功能**：HybridPositionalEncoding3D + LoRA 训练（4.5M参数）
- **特点**：
  - 使用 Cross-Attention 融合3D信息
  - 需要 DeepSpeed ZeRO-2/3
  - 显存需求高（15-20GB/卡）
- **使用**：
  ```bash
  # Stage 1: 只训3D Encoder
  python train_3d_lora.py --stage 1 --max_frames 16 --deepspeed ds_config.json
  
  # Stage 2: 加LoRA训LLM
  python train_3d_lora.py --stage 2 --checkpoint ./checkpoints/stage1/best.pt
  ```

### 数据预处理脚本

#### `extract_frames_for_3d.py`
- **功能**：从 VSI-Bench 视频提取均匀采样的帧
- **输出**：`/mnt/data/qyk/43d_16f/` 等目录
- **使用**：
  ```bash
  python extract_frames_for_3d.py \
      --dataset all \
      --output_base /mnt/data/qyk/43d_16f \
      --num_frames 16
  ```

#### `extract_scannet_frames.py`
- **功能**：从 ScanNet 提取帧（跨数据集验证用）
- **说明**：ScanNet 数据需另行下载
- **使用**：
  ```bash
  python extract_scannet_frames.py \
      --scannet_root /mnt/data/scannet/scans \
      --output_dir /mnt/data/scannet_frames_16f \
      --num_frames 16
  ```

### 测试与工具脚本

#### `test_3d_encoding.py`
- **功能**：测试3D编码流程，验证输出无 NaN/Inf
- **使用**：`python test_3d_encoding.py`
- **场景**：修改代码后快速验证3D模块是否正常

#### `list_scripts.py`
- **功能**：列出本项目所有脚本文件说明
- **使用**：`python list_scripts.py --detail`

---

## 🔧 Shell 脚本

### 训练启动脚本（推荐用这些）

#### `train_residual_4gpu.sh`
- **功能**：4卡 DDP 训练 Residual3D
- **GPU**：使用 0,1,2,3 四张卡
- **使用**：`bash train_residual_4gpu.sh`
- **输出**：`checkpoints/3d_residual_16f_4gpu/`

#### `train_zero2_4gpu.sh`（不用）
- **功能**：4卡 DeepSpeed ZeRO-2 训练 Hybrid3D
- **GPU**：使用 0,1,2,3 四张卡
- **使用**：`bash train_zero2_4gpu.sh`
- **注意**：显存占用高（15-20GB/卡）

#### `train_residual.sh`
- **功能**：单卡训练 Residual3D
- **使用**：`bash train_residual.sh --max_frames 16`
- **场景**：快速实验或显存受限

#### `train_zero2.sh`
- **功能**：单卡/多卡 DeepSpeed 训练 Hybrid3D
- **使用**：`bash train_zero2.sh`
- **特点**：自动检测GPU数量

### 评测脚本

#### `evaluate_3d.sh`
- **功能**：评测3D模型（Residual）
- **使用**：
  ```bash
  # Residual 3D 评测
  bash evaluate_3d.sh \
      --num_frames 16 \
      --checkpoint ./checkpoints/3d_residual_16f_4gpu/best/trainable_weights.pt \
      --num_processes 8
  
  ```
- **输出**：`logs/YYYYMMDD_3d/vsibench/`

#### `evaluate_all_in_one.sh`
- **功能**：Baseline 模型评测（无3D）
- **使用**：
  ```bash
  # 32帧基线
  bash evaluate_all_in_one.sh \
      --model llava_one_vision_qwen2_0p5b_ov_32f \
      --num_processes 8
  
  # 16帧基线（覆盖帧数）
  num_frames=16 bash evaluate_all_in_one.sh \
      --model llava_one_vision_qwen2_0p5b_ov_32f \
      --num_processes 8
  ```

---

## 🗂️ Map-Anything 目录下的脚本

### `batch_3d_pipeline.py`
- **路径**：`~/map-anything/batch_3d_pipeline.py`
- **功能**：使用 MapAnything 生成3D点云缓存
- **环境**：必须在 `mapanything` conda 环境中运行
- **使用**：
  ```bash
  conda activate mapanything
  python batch_3d_pipeline.py \
      --frame_base /mnt/data/qyk/43d_16f \
      --output_dir /mnt/data/qyk/43d3dpt_16f \
      --dataset all
  ```

---

## 🔄 典型工作流程

### 流程1：从头开始训练 Residual 3D
```bash
# 1. 提取视频帧
python extract_frames_for_3d.py --num_frames 16

# 2. 生成3D缓存（mapanything环境）
conda activate mapanything
python batch_3d_pipeline.py --frame_base /mnt/data/qyk/43d_16f --output_dir /mnt/data/qyk/43d3dpt_16f

# 3. 训练（vsibench环境）
conda activate vsibench
bash train_residual_4gpu.sh

# 4. 评测
bash evaluate_3d.sh --checkpoint ./checkpoints/3d_residual_16f_4gpu/best/trainable_weights.pt
```

### 流程2：快速实验（使用现有缓存）
```bash
# 直接训练（跳过1、2步）
bash train_residual_4gpu.sh
```

### 流程3：对比实验
```bash
# Residual 3D
bash evaluate_3d.sh --checkpoint ./checkpoints/3d_residual_16f_4gpu/best/trainable_weights.pt

# Baseline 16帧
num_frames=16 bash evaluate_all_in_one.sh --model llava_one_vision_qwen2_0p5b_ov_32f

# Baseline 32帧
bash evaluate_all_in_one.sh --model llava_one_vision_qwen2_0p5b_ov_32f
```

---

## 💡 常见问题

**Q：Residual 和 Hybrid 怎么选？**
- Residual（366K参数）：单卡可训，速度快，适合快速实验
- Hybrid（4.5M参数）：需要DeepSpeed，理论上对齐更精细

**Q：16帧和32帧怎么选？**
- 16帧：单卡/4卡可训，速度快，3D信息补偿帧数损失
- 32帧：需要多卡，适合时序理解任务

**Q：评测卡住怎么办？**
- 多进程评测后可能不自动退出，Ctrl+C 终止即可
- 结果已保存到 `logs/` 目录下

---

**最后更新**：2026-03-12
