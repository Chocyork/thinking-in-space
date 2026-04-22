#!/usr/bin/env python3
"""
单场景 3D 对比测试
简化版本：测试 3D 数据是否能正确加载和注入
"""

import sys
import torch

sys.path.insert(0, "/home/qyk/thinking-in-space")
sys.path.insert(0, "/home/qyk/map-anything")

print("=" * 60)
print("Courtyard 3D Test")
print("=" * 60)

# 1. 加载 3D 数据
print("\n[1] Loading 3D cache...")
cache = torch.load("/home/qyk/map-anything/3d_cache/courtyard_3d.pt")
print(f"  ✓ Frames: {cache['centers_3d'].shape[0]}")
print(f"  ✓ Patches: {cache['centers_3d'].shape[1]}")

# 2. 创建 3D 编码器
print("\n[2] Creating 3D position encoder...")
from pipeline_3d_vlm.pos_encoder_3d import HybridPositionalEncoding3D
encoder = HybridPositionalEncoding3D(dim=1152)
print(f"  ✓ Params: {sum(p.numel() for p in encoder.parameters()):,}")

# 3. 模拟视觉特征 + 3D 编码
print("\n[3] Testing 3D encoding...")
coords = cache['centers_3d'][:4]  # 取 4 帧
mask = cache['valid_mask'][:4]
visual = torch.randn(4, 729, 1152)

with torch.no_grad():
    enhanced, _ = encoder(visual, coords, mask)

print(f"  ✓ Input: {visual.shape}")
print(f"  ✓ Output: {enhanced.shape}")

# 4. 测试不同帧数
print("\n[4] Testing with different frame counts...")
for num_frames in [1, 4, 8, 16]:
    if num_frames <= len(cache['centers_3d']):
        c = cache['centers_3d'][:num_frames]
        m = cache['valid_mask'][:num_frames]
        v = torch.randn(num_frames, 729, 1152)
        with torch.no_grad():
            out, _ = encoder(v, c, m)
        print(f"  ✓ {num_frames} frames: {out.shape}")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("\nNext: Integrate this into lmms-eval's generate_until")
print("=" * 60)
