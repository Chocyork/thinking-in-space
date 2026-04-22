# 方案 A：单场景 3D 验证（Courtyard）

## 目标
用现有的 `courtyard_3d.pt` 快速验证 3D 编码器是否能工作。

## 当前状态 ✅
- 3D 数据：`courtyard_3d.pt` (38 frames, 729 patches)
- 3D 编码器：`HybridPositionalEncoding3D` (4.5M 参数)
- 验证通过：编码器能正常前向传播

## 测试步骤

### Step 1: 运行简化测试（已就绪）
```bash
cd /home/qyk/thinking-in-space/llava_onevision_3d
python test_simple_3d.py
```

预期输出：
```
✓ Frames: 38
✓ Patches: 729
✓ Params: 4,584,577
✓ Input: torch.Size([4, 729, 1152])
✓ Output: torch.Size([4, 729, 1152])
✓ All tests passed!
```

### Step 2: 手动对比测试（需要 GPU）

由于修改 lmms-eval 的 generate_until 需要较多代码，建议先做一个**手动对比**：

#### 2.1 准备 courtyard 视频的问题
创建一个简单的测试问题文件：

```json
// courtyard_questions.json
[
  {
    "question": "What is the main object in the center?",
    "expected": "table"
  },
  {
    "question": "How many chairs are visible?",
    "expected": "2"
  },
  {
    "question": "What is on the left side?",
    "expected": "tree"
  }
]
```

#### 2.2 运行 2D baseline
使用现有的评估脚本：

```bash
cd /home/qyk/thinking-in-space

# 创建一个临时的 courtyard task
python -c "
import json

# 创建简单的测试任务
task = {
    'dataset': 'courtyard',
    'scene_name': 'courtyard',
    'questions': [
        {
            'question': 'What is the main object in the center?',
            'ground_truth': 'table',
            'question_type': 'object_rel_direction_easy'
        },
        {
            'question': 'How many chairs are visible?',
            'ground_truth': '2',
            'question_type': 'object_counting'
        }
    ]
}

with open('courtyard_task.json', 'w') as f:
    json.dump(task, f, indent=2)

print('Task file created')
"
```

#### 2.3 运行对比
由于完整集成需要时间，建议先这样对比：

```bash
# 1. 先跑 2D baseline（已有脚本）
bash evaluate_all_in_one.sh \
    --model llava_onevision_qwen2_0p5b_ov_32f \
    --limit 1  # 只跑一个样本

# 记录结果，然后...

# 2. 手动修改 llava_onevision.py 注入 3D 数据（下一步）
```

### Step 3: 修改 llava_onevision.py 实现 3D 注入

这是关键步骤。需要修改 `/home/qyk/thinking-in-space/lmms_eval/models/llava_onevision.py`：

#### 修改点 1: __init__ 添加 3D 支持
```python
def __init__(self, ..., use_3d=False, point_cloud_path=None):
    # ... 原有代码 ...
    
    self.use_3d = use_3d
    if use_3d and point_cloud_path:
        # 加载 3D 数据
        self.data_3d = torch.load(point_cloud_path)
        
        # 初始化 3D 编码器
        from pipeline_3d_vlm.pos_encoder_3d import HybridPositionalEncoding3D
        self.pos_encoder = HybridPositionalEncoding3D(dim=1152)
        self.pos_encoder.eval()
        
        # Hook encode_images 方法
        self._original_encode_images = self.model.encode_images
        self.model.encode_images = self._encode_images_with_3d
```

#### 修改点 2: 创建 3D 编码 wrapper
```python
def _encode_images_with_3d(self, images):
    # 1. 原有 vision tower
    features = self.vision_tower(images)
    
    # 2. 注入 3D 信息
    if hasattr(self, '_current_3d_coords'):
        coords = self._current_3d_coords.to(images.device)
        mask = self._current_3d_mask.to(images.device)
        features, _ = self.pos_encoder(features, coords, mask)
    
    # 3. projector
    features = self.mm_projector(features)
    return features
```

#### 修改点 3: generate_until 中加载 3D 数据
```python
def generate_until(self, requests):
    # ... 原有代码 ...
    
    # 为每个请求加载 3D 数据
    for req in requests:
        if self.use_3d:
            # 获取视频路径
            video_path = ...
            
            # 计算帧索引
            frame_indices = self._get_frame_indices(video_path)
            
            # 加载对应的 3D 坐标
            coords = self.data_3d['centers_3d'][frame_indices]
            mask = self.data_3d['valid_mask'][frame_indices]
            
            # 设置当前 3D 数据
            self._current_3d_coords = coords
            self._current_3d_mask = mask
    
    # ... 调用原有生成逻辑 ...
```

## 简化方案：只改一行进行测试

如果不想大规模修改代码，可以先这样快速测试：

```python
# 在 llava_onevision.py 的 generate_until 方法开头添加：

if hasattr(self, 'test_3d_cache'):
    # 测试模式：直接注入 courtyard 的 3D 数据
    print("[TEST] Injecting 3D data from courtyard_3d.pt")
    import torch
    data_3d = torch.load('/home/qyk/map-anything/3d_cache/courtyard_3d.pt')
    
    # 假设当前 batch 是 courtyard 视频
    coords = data_3d['centers_3d'][:self.max_frames_num]
    mask = data_3d['valid_mask'][:self.max_frames_num]
    
    # 在这里注入到模型...
```

## 建议的行动计划

### 今天就做（30 分钟）
1. ✅ 运行 `test_simple_3d.py` 确认环境（已完成）
2. 🔄 手动准备 3-5 个 courtyard 视频的问题
3. 🔄 运行 2D baseline 记录结果

### 明天做（2 小时）
4. 修改 `llava_onevision.py` 添加 3D 注入
5. 运行对比测试
6. 记录 2D vs 3D 的差异

### 后天做
7. 如果有效，批量处理 10 个 VSI-Bench 视频（方案 B）

## 需要我现在做什么？

1. **实现完整的 evaluator_3d.py**（修改 lmms-eval 的代码）
2. **创建手动测试脚本**（不修改 lmms-eval，直接调用模型）
3. **准备 courtyard 的具体问题**（基于实际视频内容）

请选择一个，我立即实现！
