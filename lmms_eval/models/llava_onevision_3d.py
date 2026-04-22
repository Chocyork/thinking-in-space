"""
LLaVA-OneVision with 3D Spatial Encoding
========================================
继承原始 Llava_OneVision，添加 3D 位置编码支持

使用方法:
    --model llava_onevision_3d \
    --model_args pretrained=...,use_3d=True,point_cloud_path=...
"""

import sys
import torch
import numpy as np
from decord import VideoReader, cpu

sys.path.insert(0, "/home/qyk/map-anything")
sys.path.insert(0, "/home/qyk/thinking-in-space")

# 导入原始类
from lmms_eval.models.llava_onevision import Llava_OneVision
from lmms_eval.api.registry import register_model

try:
    from pipeline_3d_vlm.pos_encoder_3d import HybridPositionalEncoding3D
    HAS_3D = True
except ImportError:
    HAS_3D = False
    print("Warning: 3D modules not available")

# 导入残差 3D 注入
try:
    from lmms_eval.models.pos_encoder_3d_residual import Residual3DInjection, Residual3DInjectionV2
    HAS_RESIDUAL_3D = True
except ImportError as e:
    HAS_RESIDUAL_3D = False
    print(f"Warning: Residual 3D modules not available: {e}")


@register_model("llava_onevision_3d")
class Llava_OneVision_3D(Llava_OneVision):
    """
    LLaVA-OneVision with 3D Position Encoding
    
    新增参数:
        use_3d: 是否启用 3D 编码
        point_cloud_path: 3D 点云缓存路径
    """
    
    def __init__(
        self,
        pretrained: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        use_3d: bool = False,
        point_cloud_path: str = None,
        checkpoint_path: str = None,  # 3D Encoder 训练权重路径
        encoder_type: str = "residual",  # "cross_attn" 或 "residual"
        **kwargs
    ):
        # 调用父类初始化
        super().__init__(pretrained=pretrained, **kwargs)
        
        self.use_3d = use_3d
        self.encoder_type = encoder_type
        self.point_cloud_path = point_cloud_path
        
        if self.use_3d:
            if not point_cloud_path:
                raise ValueError("point_cloud_path is required when use_3d=True")
            
            import os
            if not os.path.isdir(point_cloud_path):
                raise ValueError(f"point_cloud_path must be a directory: {point_cloud_path}")
            
            self.point_cloud_dir = point_cloud_path
            self._3d_cache_pool = {}  # 缓存池，避免重复加载
            
            # 初始化 3D 编码器
            if encoder_type == "cross_attn":
                # 原始的 cross-attention 编码器
                if not HAS_3D:
                    raise ImportError("HybridPositionalEncoding3D not available")
                self.pos_encoder = HybridPositionalEncoding3D(
                    dim=1152,  # SigLIP 输出维度
                    coord_hidden_dim=512,
                    num_fusion_layers=2,
                    dropout=0.1
                ).to(self.device)
                print("[3D] Using cross-attention encoder (4.5M params)")
            else:
                # 残差注入编码器（默认，更轻量）
                if not HAS_RESIDUAL_3D:
                    raise ImportError("Residual3DInjection not available")
                self.pos_encoder = Residual3DInjection(
                    feature_dim=1152,
                    hidden_dim=256,
                    dropout=0.1
                ).to(self.device)
                print("[3D] Using residual injection encoder (~100K params)")
            
            # 加载训练好的权重（如果提供）
            if checkpoint_path and os.path.exists(checkpoint_path):
                self._load_3d_checkpoint(checkpoint_path)
            
            # 冻结 3D 编码器（推理模式）
            self.pos_encoder.eval()
            for param in self.pos_encoder.parameters():
                param.requires_grad = False
            
            # Hook encode_images 方法
            self._original_encode_images = self.model.encode_images
            self.model.encode_images = self._encode_images_with_3d
            
            print(f"[3D] Enabled with cache directory: {point_cloud_path}")
    
    def _load_3d_checkpoint(self, checkpoint_path: str):
        """
        加载 3D Encoder 训练权重
        
        支持两种格式:
        1. 完整 checkpoint: {'model_state_dict': {...}, ...}
        2. 仅 trainable_weights: {'pos_encoder.xxx': tensor, ...}
        """
        import os
        
        print(f"[3D] Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 如果 checkpoint 包含 model_state_dict，提取它
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'pos_encoder' in checkpoint or any(k.startswith('pos_encoder') for k in checkpoint.keys()):
            state_dict = checkpoint
        else:
            print(f"[3D] Warning: Checkpoint format not recognized, keys: {list(checkpoint.keys())[:5]}")
            return
        
        # 只加载 pos_encoder 相关的权重
        pos_encoder_state = {}
        for key, value in state_dict.items():
            if 'pos_encoder' in key:
                # 移除前缀（如果有）
                new_key = key.replace('model.pos_encoder.', 'pos_encoder.')
                new_key = new_key.replace('pos_encoder.', '')
                pos_encoder_state[new_key] = value
        
        if len(pos_encoder_state) > 0:
            try:
                self.pos_encoder.load_state_dict(pos_encoder_state, strict=True)
                print(f"[3D] Loaded {len(pos_encoder_state)} parameters into pos_encoder")
            except RuntimeError as e:
                print(f"[3D] Error loading state_dict: {e}")
                print(f"[3D] Attempting to load with strict=False...")
                self.pos_encoder.load_state_dict(pos_encoder_state, strict=False)
                print(f"[3D] Loaded with strict=False")
        else:
            print(f"[3D] Warning: No pos_encoder parameters found in checkpoint")
    
    def _load_3d_cache(self, cache_name: str):
        """
        加载 3D 点云缓存
        
        Args:
            cache_name: 缓存文件名（如 'courtyard' 或 'courtyard_3d.pt'）
        """
        import os
        
        # 移除扩展名（如果有）
        if cache_name.endswith('.pt'):
            cache_name = cache_name[:-3]
        
        # 检查缓存池
        if cache_name in self._3d_cache_pool:
            return self._3d_cache_pool[cache_name]
        
        # 构建路径
        cache_path = os.path.join(self.point_cloud_dir, f"{cache_name}.pt")
        
        if not os.path.exists(cache_path):
            print(f"[3D] Warning: Cache not found: {cache_path}")
            return None
        
        # 加载缓存
        data_3d = torch.load(cache_path, map_location='cpu')
        self._3d_cache_pool[cache_name] = data_3d
        
        print(f"[3D] Loaded cache: {cache_name} "
              f"({data_3d['centers_3d'].shape[0]} frames, "
              f"{data_3d['valid_mask'].float().mean():.1%} valid)")
        
        return data_3d
    
    def _encode_images_with_3d(self, images):
        """
        替换原有的 encode_images，加入 3D 位置编码
        """
        # 1. Vision Tower
        image_features = self.model.get_vision_tower()(images)
        if isinstance(image_features, tuple):
            image_features = image_features[0]
        
        # 2. 3D 位置编码（仅在数据存在时应用）
        if hasattr(self, '_current_3d_coords') and self._current_3d_coords is not None:
            coords = self._current_3d_coords.to(images.device)
            mask = self._current_3d_mask.to(images.device)
            
            # 保存原始 dtype
            orig_dtype = image_features.dtype
            
            # 确保维度匹配
            B, N, D = image_features.shape
            
            # 确保 mask 有正确的维度 (B, N, 1)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)  # (B, N) -> (B, N, 1)
            
            if coords.shape[0] != B or coords.shape[1] != N:
                # 需要插值匹配
                coords = self._interpolate_3d_coords(coords, N)
                mask = self._interpolate_3d_mask(mask.squeeze(-1), N)
            
            # 应用 3D 编码 (在 float32 下进行)
            image_features_float = image_features.float()
            image_features_enhanced, _ = self.pos_encoder(image_features_float, coords, mask)
            
            # 转换回原始 dtype
            image_features = image_features_enhanced.to(orig_dtype)
        
        # 3. Projector
        image_features = self.model.get_model().mm_projector(image_features)
        return image_features
    
    def _interpolate_3d_coords(self, coords, target_n):
        """插值 3D 坐标到目标 patch 数 (B, N, 3) -> (B, target_n, 3)"""
        B, N, C = coords.shape
        if N == target_n:
            return coords
        # 插值 patch 维度
        coords = coords.transpose(1, 2)  # (B, 3, N)
        coords = torch.nn.functional.interpolate(
            coords, size=target_n, 
            mode='linear', align_corners=False
        )
        coords = coords.transpose(1, 2)  # (B, target_n, 3)
        return coords
    
    def _interpolate_3d_mask(self, mask, target_n):
        """插值 mask 到目标 patch 数 (B, N) -> (B, target_n, 1)"""
        B, N = mask.shape
        if N == target_n:
            return mask.unsqueeze(-1)  # (B, N, 1)
        # 插值 patch 维度
        mask = mask.unsqueeze(1).float()  # (B, 1, N)
        mask = torch.nn.functional.interpolate(
            mask, size=target_n,
            mode='linear', align_corners=False
        )
        mask = mask.squeeze(1)  # (B, target_n)
        return (mask > 0.5).float().unsqueeze(-1)  # (B, target_n, 1)
    
    def _load_3d_for_video(self, video_path: str, num_frames: int):
        """
        为视频加载对应的 3D 数据
        
        根据视频文件名匹配对应的 3D 缓存文件
        支持 VSI-Bench 命名: {dataset}_{video_name}_3d.pt
        """
        import os
        
        # 从视频路径提取视频名称和数据集
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        parent_dir = os.path.basename(os.path.dirname(video_path))
        
        # 尝试多种命名方式
        cache_names = [
            video_name,  # 原始名
            f"{parent_dir}_{video_name}",  # {dataset}_{video_name}
            f"{video_name}_3d",  # {video_name}_3d
            f"{parent_dir}_{video_name}_3d",  # {dataset}_{video_name}_3d
        ]
        
        data_3d = None
        for cache_name in cache_names:
            data_3d = self._load_3d_cache(cache_name)
            if data_3d is not None:
                break
        
        if data_3d is None:
            print(f"[3D] Warning: No 3D cache found for video: {video_name} (tried: {cache_names})")
            return None, None
        
        num_3d_frames = data_3d['centers_3d'].shape[0]
        
        # 获取视频总帧数
        vr = VideoReader(video_path, ctx=cpu(0))
        total_video_frames = len(vr)
        
        # 计算采样的帧索引
        sample_indices = np.linspace(
            0, total_video_frames - 1, num_frames, dtype=int
        )
        
        # 映射到 3D 帧索引
        ratios = sample_indices / total_video_frames
        frame_3d_indices = (ratios * (num_3d_frames - 1)).astype(int)
        
        # 提取 3D 数据
        coords = data_3d['centers_3d'][frame_3d_indices]  # (T, 729, 3)
        mask = data_3d['valid_mask'][frame_3d_indices]    # (T, 729)
        
        return coords, mask.unsqueeze(-1)  # (T, 729, 3), (T, 729, 1)
    
    def generate_until(self, requests):
        """
        重写生成方法，注入 3D 数据
        """
        if not self.use_3d:
            return super().generate_until(requests)
        
        # 预加载 3D 数据
        if requests:
            first_req = requests[0]
            contexts, doc_to_target, doc_to_visual, doc_id, task, split = first_req.args
            visual = doc_to_visual(self.task_dict[task][split][doc_id])
            
            if visual and isinstance(visual[0], str):
                video_path = visual[0]
                coords, mask = self._load_3d_for_video(video_path, self.max_frames_num)
                
                if coords is not None:
                    # 保持原始维度 (T, N, 3)，与 vision tower 输出对齐
                    self._current_3d_coords = coords  # (T, N, 3)
                    self._current_3d_mask = mask.squeeze(-1)  # (T, N)
                    print(f"[3D] Loaded: {coords.shape}")
        
        try:
            # 调用父类生成
            results = super().generate_until(requests)
        finally:
            # 清理 3D 数据
            if hasattr(self, '_current_3d_coords'):
                delattr(self, '_current_3d_coords')
            if hasattr(self, '_current_3d_mask'):
                delattr(self, '_current_3d_mask')
        
        return results
