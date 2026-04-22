# VSI-Bench 3D 数据准备指南

## 现状分析

目前你只有 1 个视频的 3D 数据：
- ✅ `courtyard_3d.pt` - 1 个庭院场景
- ❌ VSI-Bench 需要 288 个视频的 3D 数据

## 解决方案

### 方案一：先在小样本上验证（推荐 ⭐）

**思路**：不处理全部 288 个视频，先选 5-10 个典型场景验证思路

**步骤：**

#### 1. 选择测试场景
从 VSI-Bench 中选择几种不同类型的场景：
```python
# 建议选择的场景（可以从 scannet/scannetpp/arkitscenes 各选几个）
test_scenes = [
    # ScanNet 场景
    "scene0251_00",  # 房间
    "scene0300_00",  # 厨房
    # ScanNet++ 场景
    "7e7cd69d0d",    # 办公室
    # ARKitScenes 场景
    "41048174",      # 卧室
]
```

#### 2. 提取视频帧
```bash
# 为每个场景提取视频帧
python scripts/extract_frames.py \
    --video_path /mnt/data/qyk/nyu-visionx/VSI-Bench/scannet/scene0251_00.mp4 \
    --output_dir /mnt/data/qyk/vsibench_frames/scene0251_00
```

#### 3. 批量生成 3D 缓存
```bash
# 使用你的 llava_onevision_14_square.py 批量处理
python scripts/batch_generate_3d.py \
    --input_dir /mnt/data/qyk/vsibench_frames \
    --output_dir /mnt/data/qyk/vsibench_3d_cache
```

---

### 方案二：完整数据处理（长期目标）

如果需要完整的 288 个视频的 3D 数据：

#### 步骤 1：获取 VSI-Bench 视频列表
```python
import json

# 从 meta_info 获取所有场景
scenes = []
for meta_file in ['scannet_meta_info_val.json', 'scannetpp_meta_info_val.json', 'arkitscenes_meta_info_val.json']:
    with open(f'data/meta_info/{meta_file}') as f:
        data = json.load(f)
        scenes.extend(list(data.keys()))

print(f"Total scenes: {len(scenes)}")  # 应该是 288 个
```

#### 步骤 2：批量提取帧 + 生成 3D
```python
# scripts/batch_process_vsibench.py
import os
import subprocess
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path, output_dir):
    """提取视频帧"""
    os.makedirs(output_dir, exist_ok=True)
    cmd = f"ffmpeg -i {video_path} -vf 'fps=1' {output_dir}/frame_%04d.jpg"
    subprocess.run(cmd, shell=True, check=True)

def generate_3d_cache(frame_dir, output_path):
    """生成 3D 缓存"""
    # 调用你的 llava_onevision_14_square.py 逻辑
    from llava_onevision_14_square import process_video
    process_video(frame_dir, output_path)

# 批量处理
for scene_name in tqdm(scenes):
    # 确定数据集类型
    if scene_name.startswith('scene'):
        dataset = 'scannet'
    elif len(scene_name) == 10:
        dataset = 'scannetpp'
    else:
        dataset = 'arkitscenes'
    
    video_path = f"/mnt/data/qyk/nyu-visionx/VSI-Bench/{dataset}/{scene_name}.mp4"
    frame_dir = f"/mnt/data/qyk/vsibench_frames/{scene_name}"
    cache_path = f"/mnt/data/qyk/vsibench_3d_cache/{scene_name}_3d.pt"
    
    # 跳过已处理的
    if os.path.exists(cache_path):
        continue
    
    # 提取帧
    if not os.path.exists(frame_dir):
        extract_frames(video_path, frame_dir)
    
    # 生成 3D 缓存
    generate_3d_cache(frame_dir, cache_path)
```

**预计时间：**
- 288 个视频 × 每视频 2-5 分钟 = **10-24 小时**
- 存储空间：每个 .pt 文件 ~0.5MB，总计 ~150MB

---

## 零样本测试的可行方案

### 方案 A：单场景深度测试（立即可做）

既然你只有 `courtyard` 的 3D 数据，可以：

1. **准备 courtyard 视频的问题集**
   - 手动创建几个关于 courtyard 视频的空间问题
   - 例如："桌子在沙发的左边还是右边？"

2. **对比测试**
   ```python
   # 测试代码框架
   from lmms_eval.models.llava_onevision import Llava_OneVision
   from llava_onevision_3d.evaluator_3d import Llava_OneVision_3D
   
   # 2D baseline
   model_2d = Llava_OneVision(pretrained="...")
   
   # 3D enhanced
   model_3d = Llava_OneVision_3D(
       pretrained="...",
       use_3d=True,
       point_cloud_path="/home/qyk/map-anything/3d_cache/courtyard_3d.pt"
   )
   
   # 在相同问题上测试对比
   for question in courtyard_questions:
       answer_2d = model_2d.generate(question, video_path)
       answer_3d = model_3d.generate(question, video_path)
       print(f"Q: {question}")
       print(f"2D: {answer_2d}")
       print(f"3D: {answer_3d}")
   ```

### 方案 B：部分 VSI-Bench 测试（需要准备数据）

选择 5-10 个 VSI-Bench 场景，批量生成 3D 数据后测试：

```bash
# 1. 先处理 5 个场景
python scripts/batch_process.py --num_scenes 5

# 2. 运行评估（只评估有 3D 数据的场景）
bash evaluate_all_in_one.sh \
    --model llava_onevision_3d \
    --3d_cache_dir /mnt/data/qyk/vsibench_3d_cache \
    --filter_scenes "scene0251_00,scene0300_00,..."
```

---

## 推荐行动计划

### 第一阶段（本周）：单场景验证
**目标**：验证 3D 编码器是否工作正常

1. 使用现有的 `courtyard_3d.pt`
2. 人工准备 5-10 个 courtyard 视频的空间问题
3. 对比 2D vs 3D 的回答差异
4. **如果有效** → 进入第二阶段

### 第二阶段（下周）：小批量 VSI-Bench 验证
**目标**：验证在标准 benchmark 上的效果

1. 选择 10 个 VSI-Bench 场景（不同类型）
2. 批量生成 3D 缓存
3. 在 lmms-eval 中实现 3D 数据加载
4. 对比 2D baseline vs 3D enhanced 的准确率

### 第三阶段（长期）：完整 benchmark
**目标**：完整 VSI-Bench 评估

1. 批量处理全部 288 个视频
2. 端到端微调 3D 编码器（如果需要）
3. 提交结果对比 baseline

---

## 需要我帮你实现哪个脚本？

1. **单场景测试脚本**（用 courtyard 快速验证）
2. **批量 3D 数据生成脚本**（处理多个 VSI-Bench 视频）
3. **VSI-Bench 子集评估脚本**（只评估有 3D 数据的场景）

请告诉我你的选择，我来具体实现！
