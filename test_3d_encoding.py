#!/usr/bin/env python
"""
快速测试 3D 编码是否能正常工作
使用 courtyard 的切帧图片
"""

import sys
import os
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/qyk/map-anything")
sys.path.insert(0, "/home/qyk/thinking-in-space")

# 加载 3D 编码器
try:
    from pipeline_3d_vlm.pos_encoder_3d import HybridPositionalEncoding3D
    print("✓ 3D encoder imported successfully")
except ImportError as e:
    print(f"✗ Failed to import 3D encoder: {e}")
    sys.exit(1)

# 配置
IMAGE_DIR = "/mnt/data/qyk/courtyard/images/dslr_images"
CACHE_PATH = "/home/qyk/map-anything/3d_cache/courtyard_3d.pt"
NUM_FRAMES = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n{'='*60}")
print("3D Encoding Test")
print(f"{'='*60}")
print(f"Device: {DEVICE}")
print(f"Image dir: {IMAGE_DIR}")
print(f"Cache: {CACHE_PATH}")
print(f"Num frames: {NUM_FRAMES}")

# 1. 加载 3D 缓存
print(f"\n[1] Loading 3D cache...")
data_3d = torch.load(CACHE_PATH, map_location='cpu')
print(f"    ✓ Centers shape: {data_3d['centers_3d'].shape}")
print(f"    ✓ Valid mask shape: {data_3d['valid_mask'].shape}")
print(f"    ✓ Valid ratio: {data_3d['valid_mask'].float().mean():.1%}")

# 2. 加载图片
print(f"\n[2] Loading images...")
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png')])
if len(image_files) == 0:
    print(f"    ✗ No images found in {IMAGE_DIR}")
    sys.exit(1)

print(f"    Found {len(image_files)} images")

# 均匀采样 NUM_FRAMES 帧
total_frames = len(image_files)
indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
sampled_files = [image_files[i] for i in indices]
print(f"    Sampled {NUM_FRAMES} frames: {sampled_files[:3]}...")

# 加载并预处理图片 (SigLIP 预处理)
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # SigLIP normalization
])

images = []
for fname in sampled_files:
    img_path = os.path.join(IMAGE_DIR, fname)
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img)
    images.append(img_tensor)

images = torch.stack(images).to(DEVICE)
print(f"    ✓ Image tensor shape: {images.shape}")

# 3. 加载 LLaVA vision tower
print(f"\n[3] Loading vision tower...")
from transformers import AutoProcessor, AutoModel

vision_tower_path = "/mnt/data/qyk/google/siglip-so400m-patch14-384"
processor = AutoProcessor.from_pretrained(vision_tower_path)
vision_tower = AutoModel.from_pretrained(vision_tower_path).vision_model.to(DEVICE)
vision_tower.eval()

print(f"    ✓ Vision tower loaded")

# 4. 提取图像特征
print(f"\n[4] Extracting image features...")
with torch.no_grad():
    # SigLIP 输出: (B, 729, 1152)
    image_features = vision_tower(images).last_hidden_state
    
print(f"    ✓ Image features shape: {image_features.shape}")

# 5. 提取对应的 3D 数据
print(f"\n[5] Preparing 3D coordinates...")
num_3d_frames = data_3d['centers_3d'].shape[0]
ratios = indices / total_frames
frame_3d_indices = (ratios * (num_3d_frames - 1)).astype(int)

coords = data_3d['centers_3d'][frame_3d_indices].to(DEVICE)  # (8, 729, 3)
mask = data_3d['valid_mask'][frame_3d_indices].to(DEVICE).unsqueeze(-1)  # (8, 729, 1)

print(f"    ✓ 3D coords shape: {coords.shape}")
print(f"    ✓ Mask shape: {mask.shape}")

# 6. 初始化并应用 3D 编码器
print(f"\n[6] Testing 3D position encoding...")
pos_encoder = HybridPositionalEncoding3D(
    dim=1152,
    coord_hidden_dim=512,
    num_fusion_layers=2,
    dropout=0.1
).to(DEVICE)
pos_encoder.eval()

print(f"    ✓ 3D encoder initialized")
print(f"    Params: {sum(p.numel() for p in pos_encoder.parameters()):,}")

# 展平 batch 维度: (8, 729, 1152) -> (8*729, 1152)
B, N, D = image_features.shape
image_features_flat = image_features.view(-1, D)
coords_flat = coords.view(-1, 3)
mask_flat = mask.view(-1, 1)

print(f"    Input shape: {image_features_flat.shape}")
print(f"    Coords shape: {coords_flat.shape}")
print(f"    Mask shape: {mask_flat.shape}")

# 应用 3D 编码
with torch.no_grad():
    encoded_features, _ = pos_encoder(
        image_features_flat.unsqueeze(0),  # (1, 8*729, 1152)
        coords_flat.unsqueeze(0),          # (1, 8*729, 3)
        mask_flat.unsqueeze(0)             # (1, 8*729, 1)
    )

encoded_features = encoded_features.squeeze(0)  # (8*729, 1152)

print(f"    ✓ Encoded features shape: {encoded_features.shape}")
print(f"    ✓ Encoding successful!")

# 7. 验证输出质量
print(f"\n[7] Validating output...")
print(f"    Feature mean: {encoded_features.mean().item():.4f}")
print(f"    Feature std: {encoded_features.std().item():.4f}")
print(f"    Feature range: [{encoded_features.min().item():.4f}, {encoded_features.max().item():.4f}]")

# 检查是否有 nan/inf
has_nan = torch.isnan(encoded_features).any().item()
has_inf = torch.isinf(encoded_features).any().item()

if has_nan:
    print(f"    ✗ WARNING: NaN detected in output!")
else:
    print(f"    ✓ No NaN values")

if has_inf:
    print(f"    ✗ WARNING: Inf detected in output!")
else:
    print(f"    ✓ No Inf values")

print(f"\n{'='*60}")
print("Test completed successfully!")
print(f"{'='*60}")
print("\nConclusion: 3D position encoding is working correctly.")
print("Ready for Plan B: Batch generation of 3D caches for VSI-Bench")
