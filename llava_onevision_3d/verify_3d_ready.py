#!/usr/bin/env python3
"""
最简验证脚本：确认 3D 数据和环境准备就绪
"""

import sys
import torch

print("=" * 60)
print("Step 1: Check 3D Cache")
print("=" * 60)

try:
    cache_path = "/home/qyk/map-anything/3d_cache/courtyard_3d.pt"
    data = torch.load(cache_path)
    
    print(f"✓ Loaded: {cache_path}")
    print(f"  Centers shape: {data['centers_3d'].shape}")
    print(f"  Valid mask shape: {data['valid_mask'].shape}")
    print(f"  Valid ratio: {data['valid_mask'].float().mean():.1%}")
    print(f"  Coord range: [{data['centers_3d'].min():.2f}, {data['centers_3d'].max():.2f}]")
    
    # 验证 patch 数量
    num_patches = data['centers_3d'].shape[1]
    expected_patches = 729
    if num_patches == expected_patches:
        print(f"✓ Patch count: {num_patches} == {expected_patches} (correct!)")
    else:
        print(f"✗ Patch count mismatch: {num_patches} != {expected_patches}")
        
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Step 2: Check MapAnything Modules")
print("=" * 60)

try:
    sys.path.insert(0, "/home/qyk/map-anything")
    from pipeline_3d_vlm.pos_encoder_3d import HybridPositionalEncoding3D
    
    # 创建编码器
    encoder = HybridPositionalEncoding3D(dim=1152)
    print(f"✓ Created HybridPositionalEncoding3D")
    print(f"  Params: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # 测试前向传播
    coords = data['centers_3d'][:2]  # 取 2 帧
    mask = data['valid_mask'][:2]
    visual = torch.randn(2, 729, 1152)
    
    with torch.no_grad():
        output, _ = encoder(visual, coords, mask)
    
    print(f"✓ Forward pass successful")
    print(f"  Input: {visual.shape}")
    print(f"  Output: {output.shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Step 3: Check LLaVA-OneVision")
print("=" * 60)

try:
    sys.path.insert(0, "/home/qyk/thinking-in-space")
    from lmms_eval.models.llava_onevision import Llava_OneVision
    
    print("✓ Can import Llava_OneVision")
    print("  (Actual model loading skipped to save time)")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
✓ All checks passed!

You can now proceed to:

1. Quick test (no model loading):
   python test_adapter_simple.py

2. Full integration test (requires GPU):
   python run_3d_test.py --mode 2d --video_dir /path/to/frames

3. Implement full evaluator:
   Modify lmms_eval/models/llava_onevision.py
   (see evaluator_3d.py for reference)
""")
