# 快速开始：单场景 3D 测试

## 目标
用现有的 `courtyard_3d.pt` 快速验证 3D 编码器是否工作。

## 步骤

### 1. 验证 3D 数据（1 分钟）

```bash
python3 << 'EOF'
import torch
data = torch.load("/home/qyk/map-anything/3d_cache/courtyard_3d.pt")
print(f"Frames: {data['centers_3d'].shape[0]}")
print(f"Patches: {data['centers_3d'].shape[1]}")
print(f"Valid: {data['valid_mask'].float().mean():.1%}")
print("✓ 3D data is ready!")
EOF
```

### 2. 准备测试数据

由于你只有 courtyard 的 3D 数据，我们可以：

**方案 A：人工准备 courtyard 的问题**
```json
[
  {
    "question": "What is in the center of the scene?",
    "answer": "table",
    "video": "courtyard.mp4"
  }
]
```

**方案 B：找到 VSI-Bench 中包含 courtyard 的 subset**
检查 VSI-Bench 是否有 courtyard 场景的 annotation。

### 3. 运行对比测试

由于直接修改 lmms-eval 的 generate_until 比较复杂，我建议先做一个**简化测试**：

```bash
# 1. 先跑 2D baseline（使用现有的 evaluate_all_in_one.sh）
cd /home/qyk/thinking-in-space
bash evaluate_all_in_one.sh \
    --model llava_onevision_qwen2_0p5b_ov_32f \
    --benchmark vsibench \
    --limit 10  # 只测试前10个样本

# 2. 记录结果后，我们再实现 3D 版本进行对比
```

## 实际可行的快速验证方案

### 最简验证（无需修改 lmms-eval）

直接测试 3D Adapter 是否能正确加载和运行：

```python
# test_adapter.py
import sys
sys.path.insert(0, "/home/qyk/thinking-in-space")
sys.path.insert(0, "/home/qyk/map-anything")

import torch
from lmms_eval.models.llava_onevision import Llava_OneVision
from pipeline_3d_vlm.pos_encoder_3d import HybridPositionalEncoding3D

# 1. 加载 2D 模型
model = Llava_OneVision(
    pretrained="lmms-lab/llava-onevision-qwen2-0p5b-ov",
    device="cuda:0"
)

# 2. 创建 3D 编码器
pos_encoder = HybridPositionalEncoding3D(dim=1152)
pos_encoder = pos_encoder.cuda()

# 3. 加载 3D 数据
data = torch.load("/home/qyk/map-anything/3d_cache/courtyard_3d.pt")
coords = data['centers_3d'].cuda()  # (38, 729, 3)
valid_mask = data['valid_mask'].cuda().unsqueeze(-1)  # (38, 729, 1)

print(f"Coords: {coords.shape}, range [{coords.min():.2f}, {coords.max():.2f}]")
print(f"Valid: {valid_mask.float().mean():.1%}")

# 4. 模拟视觉特征
visual_feat = torch.randn(38, 729, 1152).cuda()

# 5. 测试 3D 编码器
with torch.no_grad():
    enhanced, _ = pos_encoder(visual_feat, coords, valid_mask)

print(f"✓ 3D encoding works! Output: {enhanced.shape}")
```

如果以上代码能运行，说明 3D 编码器是正确的。

### 下一步：集成到 lmms-eval

这需要修改 `llava_onevision.py` 的 `generate_until` 方法。

核心改动点：
1. 在 `__init__` 中添加 `use_3d` 和 `cache_3d_path` 参数
2. 在 `generate_until` 中加载视频的 3D 数据
3. 在调用 `model.generate` 之前注入 3D 数据

## 建议

既然完整的集成需要时间，我建议：

1. **今天就做**：运行上面的 `test_adapter.py` 验证 3D 编码器能工作
2. **明天做**：基于验证结果，修改 `llava_onevision.py` 实现完整集成
3. **后天做**：批量生成 10 个 VSI-Bench 视频的 3D 数据，进行标准 benchmark 测试

要我帮你实现完整的 `evaluator_3d.py` 吗？这需要深入修改 lmms-eval 的代码。
