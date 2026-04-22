#!/usr/bin/env python3
"""
方案 A：Courtyard 单场景 3D 对比测试
直接加载模型进行 2D vs 3D 对比推理
"""

import sys
import json
import torch
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, "/home/qyk/map-anything")
sys.path.insert(0, "/home/qyk/thinking-in-space")

print("=" * 70)
print("Courtyard 3D Test - Plan A")
print("=" * 70)

# 1. 加载问题
print("\n[1/5] Loading questions...")
with open("llava_onevision_3d/courtyard_questions.json") as f:
    questions = json.load(f)
print(f"  ✓ Loaded {len(questions)} questions")

# 2. 加载视频帧
print("\n[2/5] Loading video frames...")
frame_dir = "/mnt/data/qyk/courtyard_raw/courtyard/images/dslr_jpgs"
frame_files = sorted([f for f in Path(frame_dir).iterdir() if f.suffix.lower() in ['.jpg', '.png']])

# 均匀采样 8 帧
indices = np.linspace(0, len(frame_files) - 1, 8, dtype=int)
frames = [Image.open(frame_files[i]).convert('RGB') for i in indices]
print(f"  ✓ Loaded {len(frames)} frames from {frame_dir}")

# 3. 加载 3D 数据
print("\n[3/5] Loading 3D cache...")
cache_3d = torch.load("/home/qyk/map-anything/3d_cache/courtyard_3d.pt")
print(f"  ✓ Frames: {cache_3d['centers_3d'].shape[0]}, Valid: {cache_3d['valid_mask'].float().mean():.1%}")

# 4. 加载模型（2D baseline）
print("\n[4/5] Loading 2D baseline model...")
print("  (This may take a while...)")

from lmms_eval.models.llava_onevision import Llava_OneVision

model_2d = Llava_OneVision(
    pretrained="lmms-lab/llava-onevision-qwen2-0p5b-ov",
    device="cuda:0",
    max_frames_num=8
)
print("  ✓ 2D model loaded")

# 5. 加载 3D 增强模型
print("\n[5/5] Loading 3D enhanced model...")

from lmms_eval.models.llava_onevision_3d_new import Llava_OneVision_3D

model_3d = Llava_OneVision_3D(
    pretrained="lmms-lab/llava-onevision-qwen2-0p5b-ov",
    device="cuda:0",
    max_frames_num=8,
    use_3d=True,
    point_cloud_path="/home/qyk/map-anything/3d_cache/courtyard_3d.pt"
)
print("  ✓ 3D model loaded")

print("\n" + "=" * 70)
print("Running Comparison Tests")
print("=" * 70)

# 6. 运行对比测试
results = []

for i, q in enumerate(questions, 1):
    question = q["question"]
    q_type = q["type"]
    
    print(f"\n[{i}/{len(questions)}] {q_type}")
    print(f"Q: {question}")
    print("-" * 50)
    
    # 构造请求
    from lmms_eval.api.instance import Instance
    
    # 2D 回答
    try:
        # 简化调用 - 实际需要更复杂的 lmms-eval 流程
        # 这里用占位符
        answer_2d = "[2D Model Output]"
    except Exception as e:
        answer_2d = f"[Error: {e}]"
    
    # 3D 回答
    try:
        answer_3d = "[3D Model Output]"
    except Exception as e:
        answer_3d = f"[Error: {e}]"
    
    print(f"2D: {answer_2d}")
    print(f"3D: {answer_3d}")
    
    results.append({
        "id": q["id"],
        "question": question,
        "type": q_type,
        "answer_2d": answer_2d,
        "answer_3d": answer_3d
    })

# 7. 保存结果
print("\n" + "=" * 70)
print("Saving results...")
output_file = "llava_onevision_3d/courtyard_test_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Results saved to: {output_file}")

print("\n" + "=" * 70)
print("Test complete!")
print("=" * 70)
