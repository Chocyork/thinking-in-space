# RunPod 迁移工作指导

## 你是谁
你是 Kimi，正在协助我在 RunPod 服务器上恢复和继续一个关于 **视觉 token 空间冗余度剪枝** 的研究项目。

## 项目背景
基于 VSI-Bench 评测框架（lmms-eval），利用 3D 重建（MapAnything / Depth-Anything-3）检测跨帧/同帧重复 patch，对 LLaVA-OneVision 的视觉 token 做空间感知剪枝。

核心发现：7B 模型在 66.5% 剪枝率下精度反超 baseline（31.43 > 31.30），说明视觉 token 存在干扰性冗余。

## RunPod 当前环境

```
/workspace/yefei/vlm3d/
├── thinking-in-space/     # 评测框架（已 git pull 最新）
│   ├── lmms_eval/models/llava_onevision_3d_*.py  # 剪枝模型文件
│   ├── extract_frames_for_3d.py                  # 提取 16 帧帧图片
│   ├── validation/                               # matching groups 数据
│   │   ├── da3_2dir/
│   │   ├── 43d3dpt_16f_match_voxel10cm_iou45/
│   │   └── 43d3dpt_16f_match_voxel30cm/
│   └── requirements_vsibench.txt / requirements_mapanything.txt
├── map-anything/          # MapAnything 3D 重建
├── Depth-Anything-3/      # DA3 3D 重建（含 step1_batch_processor.py）
└── data/
    └── 43d_16f/           # 帧图片输出目录（待生成）
    └── 43d3dpt_16f/       # 3D 缓存输出目录（待生成）
```

**Conda 环境：**
- `vlm3d_vsibench`：评测用（torch 2.10 + transformers + accelerate + deepspeed + decord）
- `vlm3d_mapanything`：3D 重建用（torch 2.9 + mapanything + depth-anything-3 + xformers + open3d）

**已下载内容：**
- ✅ 模型权重：0.5B / 7B / Video-7B（HF 缓存）
- ✅ VSI-Bench 视频：arkitscenes / scannet / scannetpp（`/workspace/.cache/huggingface/vsibench/`）
- ✅ matching_groups_core.tar.gz（GitHub Release 下载，已解压到 validation/）

**关键配置：**
- `LMMS_EVAL_LAUNCHER=python`（单卡 5090）
- `vsibench.yaml` 的 dataset_path 已改为 `nyu-visionx/VSI-Bench`（HF Hub ID）
- LLaVA-NeXT builder.py 已自动 patch（SigLIP 优先检测）

## 接下来需要做的事

### 1. 提取 16 帧帧图片
```bash
conda activate vlm3d_vsibench
cd /workspace/yefei/vlm3d/thinking-in-space
python extract_frames_for_3d.py --dataset all --num_frames 16 --output_base /workspace/yefei/vlm3d/data/43d_16f
```
- 视频源：自动检测 `/workspace/.cache/huggingface/vsibench/`
- 输出：每个视频一个子目录，内含 16 张 `frame_*.jpg`
- 这一步必须和 VLM 推理的 `max_frames_num=16` 严格对齐

### 2. 生成 3D 点云缓存
使用 Depth-Anything-3 的 batch processor：
```bash
conda activate vlm3d_mapanything
cd /workspace/yefei/vlm3d/Depth-Anything-3
python step1_batch_processor.py \
    --frame_base /workspace/yefei/vlm3d/data/43d_16f \
    --output_dir /workspace/yefei/vlm3d/data/43d3dpt_16f \
    --dataset all
```
- 使用 DA3NESTED-GIANT-LARGE 模型（首次运行会自动下载）
- 生成的 `.pt` 文件供 MapAnything 匹配算法使用

### 3. 生成 Matching Groups
使用 map-anything 的匹配算法对 3D 缓存做跨帧 patch 匹配：
```bash
conda activate vlm3d_mapanything
cd /workspace/yefei/vlm3d/map-anything
# 运行匹配脚本（具体命令需根据 map-anything 的文档调整）
```

### 4. 运行评测
```bash
conda activate vlm3d_vsibench
export LMMS_EVAL_LAUNCHER=python
cd /workspace/yefei/vlm3d/thinking-in-space

# 示例：7B prune_max + voxel10cm
python -m lmms_eval \
    --model llava_onevision_3d_prune_max \
    --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,max_frames_num=16,mm_spatial_pool_stride=2,matching_groups_path=/workspace/yefei/vlm3d/thinking-in-space/validation/43d3dpt_16f_match_voxel10cm_iou45,alpha=0.4 \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --output_path /workspace/yefei/vlm3d/logs/vsibench_7b_prune_max
```

## 关键注意事项

1. **帧对齐**：extract_frames_for_3d.py 的 `num_frames` 必须和 VLM 的 `max_frames_num` 完全一致（当前用 16）
2. **坐标映射**：原始 MapAnything 输出 27x27 grid，需映射到 LLaVA 池化后的新 grid（约 13x13）。核心公式在模型文件的 `_setup_prepare_wrapper()` 中
3. **同步切片**：剪枝时必须同步裁剪 embeds/input_ids/labels/position_ids/attention_mask，保持序列长度一致
4. **环境隔离**：vsibench 和 mapanything 是两个独立的 conda 环境，torch 版本不同，不要混用

## 当前状态检查清单

- [ ] extract_frames_for_3d.py 成功跑完，43d_16f 下有帧图片
- [ ] step1_batch_processor.py 成功跑完，43d3dpt_16f 下有 .pt 文件
- [ ] matching groups 重新生成（或验证旧的 validation/ 数据可用）
- [ ] 运行 baseline 评测确认环境正常
- [ ] 运行 prune_max / smooth_max_v2 等实验

## 参考文件

- `setup_runpod.sh`：路径适配 + SigLIP patch 脚本（已跑过）
- `requirements_vsibench.txt` / `requirements_mapanything.txt`：环境依赖
- `evaluate_test/llava_onevision/*.sh`：评测脚本模板（路径已自动替换）
- 旧服务器实验记录：`3D_SMOOTH_EXPERIMENT_LOG.txt`（在 thinking-in-space 根目录）
