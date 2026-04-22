# LLaVA-OneVision + 3D Spatial Encoding

将 MapAnything 3D 点云信息集成到 LLaVA-OneVision 模型中，增强视觉空间理解能力。

## 核心问题：是否需要微调？

### 方案一：冻结模式（推荐 ⭐）
```python
freeze_base = True
```

**做法：**
- 冻结 LLaVA-OneVision 所有参数（Vision Tower + Projector + LLM）
- 只训练 3D 位置编码器（~3.6M 参数）

**优点：**
- ✅ 不破坏预训练权重
- ✅ 训练速度快（7B 模型只需训练 3.6M 参数）
- ✅ 需要的训练数据少（VSI-Bench 几百个样本可能就够了）
- ✅ 可以验证 3D 信息的有效性

**适用场景：**
- 快速验证 idea
- 计算资源有限
- 训练数据较少

---

### 方案二：端到端微调
```python
freeze_base = False
```

**做法：**
- 解锁所有参数，联合训练

**缺点：**
- ❌ 需要大量训练数据（否则过拟合）
- ❌ 计算成本高（7B 参数训练）
- ❌ 可能破坏预训练的视觉-语言对齐

**适用场景：**
- 有大规模 3D-视频-文本数据集
- 充足的计算资源（A100 x 8）

---

## 快速开始

### 1. 准备 3D 数据

确保你已经生成了 3D 缓存：
```bash
# 在 map-anything 目录下
python llava_onevision_14_square.py
# 输出：3d_cache/courtyard_3d.pt
```

### 2. 推理（零样本测试）

```python
from llava_onevision_3d.evaluator_3d import Llava_OneVision_3D

# 初始化 3D 增强模型
model = Llava_OneVision_3D(
    pretrained="lmms-lab/llava-onevision-qwen2-7b-ov",
    use_3d=True,
    point_cloud_path="/home/qyk/map-anything/3d_cache/courtyard_3d.pt",
    freeze_base=True,  # 推理时无影响
)

# 正常推理（会自动加载 3D 数据）
results = model.generate_until(requests)
```

### 3. 训练 3D 编码器（可选）

如果你想在 VSI-Bench 上微调 3D 编码器：

```python
# train_3d_encoder.py
from llava_onevision_3d.adapter import LLaVA3DAdapter
from llava_onevision_3d.config import get_freeze_config

config = get_freeze_config()

# 初始化 adapter
adapter = LLaVA3DAdapter(
    base_model=llava_model,
    freeze_mode=config.freeze_base,
    **config.pos_encoder_kwargs
)

# 只优化 3D 编码器参数
optimizer = torch.optim.AdamW(
    adapter.get_trainable_params(),
    lr=config.lr_3d_encoder,
    weight_decay=config.weight_decay
)

# 训练循环
for epoch in range(config.num_epochs):
    for batch in dataloader:
        images, coords_3d, valid_mask, questions = batch
        
        # 设置 3D 数据
        adapter.set_3d_data(coords_3d, valid_mask)
        
        # 前向传播
        outputs = llava_model.generate(images=images, ...)
        
        # 计算损失
        loss = compute_loss(outputs, answers)
        
        # 反向传播（只更新 3D 编码器）
        loss.backward()
        optimizer.step()
        
        # 清理
        adapter.clear_3d_data()

# 保存 3D 编码器权重
adapter.save_3d_encoder("3d_encoder_best.pth")
```

---

## 架构说明

```
Input Video Frames (T, 3, 384, 384)
           ↓
    Vision Tower (SigLIP)
           ↓
    Visual Features (T, 729, 1152)
           ↓
    ┌─────────────────────────────────────┐
    │  3D Position Encoder                │
    │  - coords_3d: (T, 729, 3)           │
    │  - valid_mask: (T, 729, 1)          │
    │  - output: (T, 729, 1152)           │
    └─────────────────────────────────────┘
           ↓
    Enhanced Features (T, 729, 1152)
           ↓
    MM Projector
           ↓
    LLM Input (T, 729, 4096)
           ↓
    LLM (Qwen2-7B)
           ↓
    Generated Text
```

---

## 文件说明

| 文件 | 功能 |
|------|------|
| `adapter.py` | 3D Adapter 核心实现 |
| `evaluator_3d.py` | 修改后的评估器（继承 llava_onevision） |
| `config.py` | 配置管理 |
| `README.md` | 本文档 |

---

## 常见问题

**Q: 冻结模式下，3D 编码器需要多少训练数据？**
A: 3D 编码器只有 3.6M 参数，理论上几百个样本就可以微调。建议先用 VSI-Bench 的小子集测试。

**Q: 可以直接使用 courtyard_3d.pt 吗？**
A: 可以！patch 对齐验证已通过（729 patches，坐标范围 [-1, 1]）。

**Q: 训练需要多少显存？**
A: 冻结模式下与原始 LLaVA 相同（约 20GB for 7B 模型）。

**Q: 如何评估效果？**
A: 对比 baseline（2D 模式）和 3D 增强模式在 VSI-Bench 上的准确率。
