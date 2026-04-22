"""
LLaVA-OneVision 3D 集成 Patch
==============================

这个脚本会生成需要添加到 llava_onevision.py 的代码片段

使用方法:
1. 复制下面的代码片段
2. 粘贴到 /home/qyk/thinking-in-space/lmms_eval/models/llava_onevision.py
3. 运行测试
"""

PATCH_CODE = '''
# ==================== 3D SUPPORT (ADD THESE LINES) ====================
# 在文件开头 import 部分添加:
import sys
sys.path.insert(0, "/home/qyk/map-anything")
try:
    from pipeline_3d_vlm.pos_encoder_3d import HybridPositionalEncoding3D
    HAS_3D = True
except ImportError:
    HAS_3D = False


# ==================== 在 __init__ 方法中添加 ====================
# 在 super().__init__() 之后添加:

# 3D support
self.use_3d = kwargs.get('use_3d', False)
self.point_cloud_path = kwargs.get('point_cloud_path', None)

if self.use_3d and HAS_3D and self.point_cloud_path:
    import torch
    self.data_3d = torch.load(self.point_cloud_path)
    self.pos_encoder = HybridPositionalEncoding3D(
        dim=1152,
        coord_hidden_dim=512,
        num_fusion_layers=2,
        dropout=0.1
    ).to(self.device)
    self.pos_encoder.eval()
    
    # 冻结 3D 编码器（推理模式）
    for param in self.pos_encoder.parameters():
        param.requires_grad = False
    
    # 包装 encode_images
    self._original_encode_images = self.model.encode_images
    self.model.encode_images = self._encode_images_with_3d
    
    print(f"[3D] Enabled with cache: {self.point_cloud_path}")
    print(f"[3D] Frames: {self.data_3d['centers_3d'].shape[0]}, "
          f"Valid: {self.data_3d['valid_mask'].float().mean():.1%}")


# ==================== 在类中添加新方法 ====================

def _encode_images_with_3d(self, images):
    """
    替换原有的 encode_images，加入 3D 位置编码
    """
    # 1. Vision Tower
    image_features = self.get_model().get_vision_tower()(images)
    if isinstance(image_features, tuple):
        image_features = image_features[0]
    
    # 2. 3D 位置编码
    if hasattr(self, '_current_3d_coords'):
        coords = self._current_3d_coords.to(images.device)
        mask = self._current_3d_mask.to(images.device)
        
        # 确保维度匹配
        B, N, D = image_features.shape
        if coords.shape[1] != N:
            coords = self._interpolate_3d_coords(coords, N)
            mask = self._interpolate_3d_mask(mask, N)
        
        # 应用 3D 编码
        image_features, _ = self.pos_encoder(image_features, coords, mask)
    
    # 3. Projector
    image_features = self.get_model().mm_projector(image_features)
    return image_features


def _interpolate_3d_coords(self, coords, target_n):
    """插值 3D 坐标到目标 patch 数"""
    B, N, C = coords.shape
    if N == target_n:
        return coords
    coords = coords.transpose(1, 2)
    coords = torch.nn.functional.interpolate(
        coords.unsqueeze(-1), size=(target_n, 1), 
        mode='bilinear', align_corners=False
    )
    coords = coords.squeeze(-1).transpose(1, 2)
    return coords


def _interpolate_3d_mask(self, mask, target_n):
    """插值 mask 到目标 patch 数"""
    B, N, C = mask.shape
    if N == target_n:
        return mask
    mask = mask.transpose(1, 2).float()
    mask = torch.nn.functional.interpolate(
        mask.unsqueeze(-1), size=(target_n, 1),
        mode='bilinear', align_corners=False
    )
    mask = mask.squeeze(-1).transpose(1, 2)
    return (mask > 0.5).float()


def _load_3d_for_video(self, video_path, num_frames):
    """
    为视频加载对应的 3D 数据
    
    简化版本：假设视频名和 3D 缓存名对应
    实际使用时可能需要更复杂的映射逻辑
    """
    if not hasattr(self, 'data_3d'):
        return None, None
    
    # 从 video_path 推断 scene_name
    # 例如: /path/to/courtyard.mp4 -> courtyard
    import numpy as np
    from decord import VideoReader, cpu
    
    # 获取视频总帧数
    vr = VideoReader(video_path, ctx=cpu(0))
    total_video_frames = len(vr)
    
    # 计算采样的帧索引
    sample_indices = np.linspace(
        0, total_video_frames - 1, num_frames, dtype=int
    )
    
    # 映射到 3D 帧索引
    total_3d_frames = self.data_3d['centers_3d'].shape[0]
    ratios = sample_indices / total_video_frames
    frame_3d_indices = (ratios * (total_3d_frames - 1)).astype(int)
    
    # 提取 3D 数据
    coords = self.data_3d['centers_3d'][frame_3d_indices]  # (T, 729, 3)
    mask = self.data_3d['valid_mask'][frame_3d_indices]    # (T, 729)
    
    return coords, mask.unsqueeze(-1)  # (T, 729, 3), (T, 729, 1)


# ==================== 修改 generate_until 方法 ====================
# 在 generate_until 方法的开头添加:

# 如果是视频任务且启用了 3D，预加载 3D 数据
if self.use_3d and requests:
    # 获取第一个请求的视频路径
    first_req = requests[0]
    contexts, doc_to_target, doc_to_visual, doc_id, task, split = first_req.args
    visual = doc_to_visual(self.task_dict[task][split][doc_id])
    
    if visual and isinstance(visual[0], str):
        video_path = visual[0]
        coords, mask = self._load_3d_for_video(video_path, self.max_frames_num)
        
        if coords is not None:
            # 展平帧维度以匹配 vision tower 输出
            # (T, 729, 3) -> (T*729, 3) 或保持 (T, 729, 3) 取决于处理逻辑
            self._current_3d_coords = coords
            self._current_3d_mask = mask
            print(f"[3D] Loaded coords: {coords.shape}, valid: {mask.float().mean():.1%}")


# 在 generate_until 方法的结尾（return 之前）添加:

# 清理 3D 数据
if hasattr(self, '_current_3d_coords'):
    delattr(self, '_current_3d_coords')
    delattr(self, '_current_3d_mask')
'''


if __name__ == "__main__":
    print("=" * 70)
    print("LLaVA-OneVision 3D Integration Patch")
    print("=" * 70)
    print()
    print("This script generates the code needed to add 3D support.")
    print()
    print("Instructions:")
    print("1. Open: /home/qyk/thinking-in-space/lmms_eval/models/llava_onevision.py")
    print("2. Add the import statements at the top")
    print("3. Add __init__ code after super().__init__()")
    print("4. Add new methods to the class")
    print("5. Modify generate_until method")
    print()
    print("=" * 70)
    print("PATCH CODE:")
    print("=" * 70)
    print(PATCH_CODE)
    print()
    print("=" * 70)
    print("Alternative: Direct file modification")
    print("=" * 70)
    print()
    print("Or run this command to create a backup and apply the patch:")
    print("  python apply_patch.py")
