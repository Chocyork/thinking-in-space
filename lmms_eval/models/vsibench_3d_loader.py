"""
VSI-Bench 3D Data Loader for MapAnything Point Cloud Integration
将 MapAnything 生成的 3D 点云坐标对齐到 LLaVA-OneVision 的 patch 结构
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PointCloud3DConfig:
    """3D点云配置"""
    # MapAnything 原始重建参数
    orig_width: int = 448  # 重建输出宽度
    orig_height: int = 294  # 重建输出高度
    orig_patches_w: int = 32  # 32 patches horizontally
    orig_patches_h: int = 21  # 21 patches vertically
    
    # LLaVA-OneVision (SigLIP) 目标参数
    target_image_size: int = 384
    target_patch_size: int = 14
    target_patches_per_side: int = 27  # 384 // 14
    
    # 深度不连续阈值 (米)
    depth_discontinuity_threshold: float = 2.0
    
    # 坐标归一化方法: 'global' 或 'per_frame'
    normalize_method: str = 'global'
    
    # 全局归一化边界 (仅在 normalize_method='global' 时使用)
    global_coord_min: Optional[np.ndarray] = None  # shape (3,)
    global_coord_max: Optional[np.ndarray] = None  # shape (3,)
    
    @property
    def num_target_patches(self) -> int:
        return self.target_patches_per_side ** 2  # 729


class MapAnything3DLoader:
    """
    加载和对齐 MapAnything 3D 点云数据到 LLaVA-OneVision 的 patch 结构
    
    数据文件结构:
        {scene_name}_patch_point_sets.npz  # 每个 patch 的完整 3D 点集
        {scene_name}_patch_centers.npz     # 每个 patch 的 3D 中心坐标
        {scene_name}_dense_pointcloud.npz  # 全局点云 (可选，用于全局归一化)
    """
    
    def __init__(
        self, 
        point_cloud_dir: str,
        config: Optional[PointCloud3DConfig] = None
    ):
        self.point_cloud_dir = Path(point_cloud_dir)
        self.config = config or PointCloud3DConfig()
        
        # 缓存全局归一化参数
        if self.config.normalize_method == 'global':
            self._init_global_normalization()
    
    def _init_global_normalization(self):
        """从所有场景的全局点云计算归一化参数"""
        if self.config.global_coord_min is not None and \
           self.config.global_coord_max is not None:
            return
        
        all_points = []
        for npz_file in self.point_cloud_dir.glob("*_dense_pointcloud.npz"):
            data = np.load(npz_file)
            points = data['arr_0']  # (N, 3) or (N, 6) with colors
            if points.shape[1] >= 3:
                all_points.append(points[:, :3])
        
        if all_points:
            all_points = np.vstack(all_points)
            self.config.global_coord_min = all_points.min(axis=0)
            self.config.global_coord_max = all_points.max(axis=0)
            print(f"[3D Loader] Global coord range: {self.config.global_coord_min} to {self.config.global_coord_max}")
        else:
            # 使用默认值
            self.config.global_coord_min = np.array([-5.0, -5.0, -1.0])
            self.config.global_coord_max = np.array([5.0, 5.0, 3.0])
    
    def load_scene_3d(
        self, 
        scene_name: str,
        frame_indices: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        加载单个场景的 3D 点云数据
        
        Args:
            scene_name: 场景名称 (e.g., "scene_001")
            frame_indices: 需要加载的帧索引列表，None 表示加载所有帧
            
        Returns:
            Dict with keys:
                - 'patch_centers': (num_frames, 729, 3) 对齐后的 3D 中心坐标
                - 'valid_mask': (num_frames, 729) 有效 patch 掩码
                - 'coord_range': (2, 3) 用于反归一化的坐标范围
        """
        # 1. 加载原始数据
        centers_path = self.point_cloud_dir / f"{scene_name}_patch_centers.npz"
        pointsets_path = self.point_cloud_dir / f"{scene_name}_patch_point_sets.npz"
        
        if not centers_path.exists():
            raise FileNotFoundError(f"3D data not found: {centers_path}")
        
        centers_data = np.load(centers_path)
        patch_centers = centers_data['arr_0']  # (num_frames, 672, 3)
        
        # 2. 选择指定帧
        if frame_indices is not None:
            patch_centers = patch_centers[frame_indices]
        
        num_frames = patch_centers.shape[0]
        
        # 3. 重采样到目标 patch 数量 (27x27=729)
        aligned_centers = self._align_patches_to_target(patch_centers)
        # aligned_centers: (num_frames, 729, 3)
        
        # 4. 计算有效 mask (基于深度不连续性)
        valid_mask = self._compute_valid_mask(pointsets_path, frame_indices)
        # valid_mask: (num_frames, 729)
        
        # 5. 坐标归一化到 [-1, 1]
        normalized_centers, coord_range = self._normalize_coords(aligned_centers)
        
        return {
            'patch_centers': torch.from_numpy(normalized_centers).float(),
            'valid_mask': torch.from_numpy(valid_mask).bool(),
            'coord_range': torch.from_numpy(coord_range).float(),
            'num_frames': num_frames
        }
    
    def _align_patches_to_target(
        self, 
        patch_centers: np.ndarray
    ) -> np.ndarray:
        """
        将 32x21=672 patches 重采样到 27x27=729 patches
        
        Args:
            patch_centers: (num_frames, 672, 3) 原始坐标
        Returns:
            aligned_centers: (num_frames, 729, 3) 对齐后坐标
        """
        num_frames = patch_centers.shape[0]
        cfg = self.config
        
        # 重塑为 2D grid: (num_frames, 21, 32, 3)
        centers_2d = patch_centers.reshape(
            num_frames, 
            cfg.orig_patches_h, 
            cfg.orig_patches_w, 
            3
        )
        
        # 转换为 torch tensor 用于 interpolate
        # 从 (N, H, W, C) -> (N, C, H, W)
        centers_tensor = torch.from_numpy(centers_2d).float().permute(0, 3, 1, 2)
        
        # 使用双线性插值重采样到 27x27
        # 注意：需要 align_corners=False 以保持相对位置
        aligned = F.interpolate(
            centers_tensor,
            size=(cfg.target_patches_per_side, cfg.target_patches_per_side),
            mode='bilinear',
            align_corners=False
        )  # (num_frames, 3, 27, 27)
        
        # 重塑回 (num_frames, 729, 3)
        aligned = aligned.permute(0, 2, 3, 1).reshape(
            num_frames, 
            cfg.num_target_patches, 
            3
        )
        
        return aligned.numpy()
    
    def _compute_valid_mask(
        self,
        pointsets_path: Path,
        frame_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        基于深度不连续性计算有效 patch mask
        
        规则：如果 patch 内点的深度范围 > threshold，则标记为无效
        """
        cfg = self.config
        
        if not pointsets_path.exists():
            # 如果没有 point sets，假设全部有效
            num_frames = len(frame_indices) if frame_indices else 100  # 默认值
            return np.ones((num_frames, cfg.num_target_patches), dtype=bool)
        
        pointsets_data = np.load(pointsets_path)
        patch_point_sets = pointsets_data['arr_0']  # (num_frames, 672) object array
        
        if frame_indices is not None:
            patch_point_sets = patch_point_sets[frame_indices]
        
        num_frames = patch_point_sets.shape[0]
        
        # 计算每个原始 patch 的深度范围
        # patch_point_sets[i, j] 是一个 (N, 3) 或 (N, 6) 的数组
        orig_valid_mask = np.zeros((num_frames, cfg.orig_patches_h * cfg.orig_patches_w), dtype=bool)
        
        for i in range(num_frames):
            for j in range(cfg.orig_patches_h * cfg.orig_patches_w):
                points = patch_point_sets[i, j]
                if isinstance(points, np.ndarray) and len(points) > 0:
                    # 计算深度范围 (Z 轴)
                    depth_range = np.ptp(points[:, 2]) if points.shape[1] >= 3 else 0
                    orig_valid_mask[i, j] = depth_range < cfg.depth_discontinuity_threshold
                else:
                    orig_valid_mask[i, j] = False
        
        # 将 672 的 valid mask 重采样到 729
        # 方法：reshape 到 2D，插值，然后阈值
        valid_mask_2d = orig_valid_mask.reshape(num_frames, cfg.orig_patches_h, cfg.orig_patches_w)
        valid_tensor = torch.from_numpy(valid_mask_2d).float().unsqueeze(1)  # (N, 1, 21, 32)
        
        aligned_valid = F.interpolate(
            valid_tensor,
            size=(cfg.target_patches_per_side, cfg.target_patches_per_side),
            mode='bilinear',
            align_corners=False
        )  # (N, 1, 27, 27)
        
        # 使用 0.5 作为阈值：如果插值后 > 0.5，认为有效
        aligned_valid = (aligned_valid.squeeze(1) > 0.5).numpy()
        
        return aligned_valid.reshape(num_frames, cfg.num_target_patches)
    
    def _normalize_coords(
        self, 
        coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 3D 坐标归一化到 [-1, 1]
        
        Returns:
            normalized_coords: (num_frames, 729, 3)
            coord_range: (2, 3) [min, max] 用于反归一化
        """
        cfg = self.config
        
        if cfg.normalize_method == 'global':
            coord_min = cfg.global_coord_min
            coord_max = cfg.global_coord_max
        else:
            # 每帧独立归一化
            coord_min = coords.min(axis=1, keepdims=True)  # (num_frames, 1, 3)
            coord_max = coords.max(axis=1, keepdims=True)
        
        # 归一化到 [-1, 1]
        normalized = 2 * (coords - coord_min) / (coord_max - coord_min + 1e-8) - 1
        normalized = np.clip(normalized, -1, 1)
        
        if cfg.normalize_method == 'global':
            coord_range = np.stack([coord_min, coord_max], axis=0)
        else:
            # 对于 per-frame，返回每帧的范围
            coord_range = np.stack([
                coord_min.squeeze(1),  # (num_frames, 3)
                coord_max.squeeze(1)
            ], axis=1)  # (num_frames, 2, 3)
        
        return normalized, coord_range


class VSI Bench3DDataCollator:
    """
    用于 batch 处理的 data collator
    处理变长视频帧数的情况
    """
    
    def __init__(self, pad_to_multiple_of: int = 8):
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: List of dicts from load_scene_3d
        Returns:
            Batched and padded tensors
        """
        batch_size = len(features)
        max_frames = max(f['num_frames'] for f in features)
        
        # 对齐到 8 的倍数 (为了效率)
        if self.pad_to_multiple_of > 1:
            max_frames = ((max_frames + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        num_patches = features[0]['patch_centers'].shape[1]  # 729
        
        # 初始化 padded tensors
        batch_centers = torch.zeros(batch_size, max_frames, num_patches, 3)
        batch_valid_mask = torch.zeros(batch_size, max_frames, num_patches, dtype=torch.bool)
        batch_frame_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)
        
        for i, feat in enumerate(features):
            n_frames = feat['num_frames']
            batch_centers[i, :n_frames] = feat['patch_centers']
            batch_valid_mask[i, :n_frames] = feat['valid_mask']
            batch_frame_mask[i, :n_frames] = True
        
        return {
            'patch_centers': batch_centers,
            'valid_mask': batch_valid_mask,
            'frame_mask': batch_frame_mask,
            'max_frames': max_frames
        }


# ==================== 集成到 VSI-Bench 评测流程 ====================

def create_vsibench_3d_loader(
    point_cloud_dir: str,
    **kwargs
) -> MapAnything3DLoader:
    """
    工厂函数：创建 VSI-Bench 3D 数据加载器
    
    Usage:
        loader = create_vsibench_3d_loader(
            point_cloud_dir="/path/to/mapanything/output",
            depth_discontinuity_threshold=2.0,
            normalize_method='global'
        )
        
        # 加载单个场景
        data_3d = loader.load_scene_3d(
            scene_name="scene_001",
            frame_indices=[0, 5, 10, 15]  # 与视频采样帧对应
        )
    """
    config = PointCloud3DConfig(**kwargs)
    return MapAnything3DLoader(point_cloud_dir, config)


# 示例：修改 llava_onevision.py 以支持 3D
def example_modification_for_llava_onevision():
    """
    这是需要添加到 llava_onevision.py 的示例代码
    """
    code = '''
    # 在 Llava_OneVision.__init__ 中添加:
    def __init__(self, ..., use_3d_pos=False, point_cloud_dir=None, **kwargs):
        # ... 原有初始化 ...
        
        self.use_3d_pos = use_3d_pos
        if self.use_3d_pos and point_cloud_dir:
            from .vsibench_3d_loader import create_vsibench_3d_loader
            self.point_cloud_loader = create_vsibench_3d_loader(point_cloud_dir)
    
    # 在 generate_until 中加载 3D 数据:
    def generate_until(self, requests):
        # ... 原有代码 ...
        
        if self.use_3d_pos and task == "vsibench":
            scene_name = doc.get("scene_name")
            # 获取视频帧索引 (根据 max_frames_num 均匀采样)
            video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
            total_frames = get_video_frame_count(video_path)  # 需要实现
            frame_indices = np.linspace(
                0, total_frames - 1, 
                self.max_frames_num, 
                dtype=int
            ).tolist()
            
            # 加载 3D 坐标
            data_3d = self.point_cloud_loader.load_scene_3d(
                scene_name, frame_indices
            )
            
            # 传递给模型
            gen_kwargs["patch_centers_3d"] = data_3d["patch_centers"].to(self.device)
            gen_kwargs["valid_mask"] = data_3d["valid_mask"].to(self.device)
    '''
    return code


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("VSI-Bench 3D Data Loader Test")
    print("=" * 60)
    
    # 创建模拟数据
    test_dir = Path("./test_3d_data")
    test_dir.mkdir(exist_ok=True)
    
    # 模拟 MapAnything 输出
    num_frames = 38
    patch_centers = np.random.randn(num_frames, 672, 3).astype(np.float32)
    np.savez(test_dir / "test_scene_patch_centers.npz", patch_centers)
    
    # 模拟 point sets
    patch_point_sets = np.empty((num_frames, 672), dtype=object)
    for i in range(num_frames):
        for j in range(672):
            # 随机点数，模拟深度范围
            n_points = np.random.randint(10, 100)
            points = np.random.randn(n_points, 3).astype(np.float32)
            # 添加深度变化
            points[:, 2] *= np.random.uniform(0.5, 3.0)
            patch_point_sets[i, j] = points
    np.savez(test_dir / "test_scene_patch_point_sets.npz", patch_point_sets)
    
    # 测试加载器
    loader = create_vsibench_3d_loader(
        point_cloud_dir=str(test_dir),
        normalize_method='per_frame'
    )
    
    # 加载并检查
    frame_indices = [0, 5, 10, 15, 20, 25, 30, 35]
    data = loader.load_scene_3d("test_scene", frame_indices)
    
    print(f"\\nLoaded 3D data:")
    print(f"  Patch centers shape: {data['patch_centers'].shape}")  # (8, 729, 3)
    print(f"  Valid mask shape: {data['valid_mask'].shape}")  # (8, 729)
    print(f"  Valid patch ratio: {data['valid_mask'].float().mean():.2%}")
    print(f"  Coord range: {data['coord_range']}")
    
    # 测试 collator
    collator = VSI Bench3DDataCollator()
    batch = collator([data, data])
    print(f"\\nBatched data:")
    print(f"  Patch centers: {batch['patch_centers'].shape}")  # (2, 8, 729, 3)
    print(f"  Frame mask: {batch['frame_mask'].sum(dim=1)}")  # [8, 8]
    
    # 清理
    import shutil
    shutil.rmtree(test_dir)
    
    print("\\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)
