"""
LLaVA-OneVision 3D 评估器
修改自 lmms_eval/models/llava_onevision.py
支持加载 3D 数据并注入模型
"""

import sys
import copy
import torch
from pathlib import Path

sys.path.insert(0, "/home/qyk/map-anything")
sys.path.insert(0, "/home/qyk/thinking-in-space")

from lmms_eval.models.llava_onevision import Llava_OneVision
from lmms_eval.api.registry import register_model

from adapter import LLaVA3DAdapter


@register_model("llava_onevision_3d")
class Llava_OneVision_3D(Llava_OneVision):
    """
    LLaVA-OneVision with 3D Spatial Encoding
    
    新增参数:
        use_3d: 是否启用 3D 编码
        point_cloud_path: 3D 点云缓存文件路径
        freeze_base: 是否冻结基础模型
    """
    
    def __init__(
        self,
        pretrained: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        use_3d: bool = True,
        point_cloud_path: str = None,
        freeze_base: bool = True,
        **kwargs
    ):
        # 调用父类初始化
        super().__init__(pretrained=pretrained, **kwargs)
        
        self.use_3d = use_3d
        
        if self.use_3d and point_cloud_path:
            # 加载 3D 数据
            self._load_3d_cache(point_cloud_path)
            
            # 创建 3D Adapter
            self.adapter = LLaVA3DAdapter(
                base_model=self.model,
                freeze_mode=freeze_base,
                coord_dim=1152,
                coord_hidden_dim=512,
                num_fusion_layers=2,
                dropout=0.1
            )
            
            # 替换 encode_images 方法
            self._original_encode_images = self.model.encode_images
            self.model.encode_images = self.adapter.encode_images_with_3d
            
            print(f"[Llava3D] 3D encoding enabled with freeze_base={freeze_base}")
        else:
            print("[Llava3D] Running in 2D mode")
    
    def _load_3d_cache(self, path: str):
        """加载 3D 点云缓存"""
        if not Path(path).exists():
            raise FileNotFoundError(f"3D cache not found: {path}")
        
        data = torch.load(path)
        self.centers_3d = data['centers_3d']      # (N, 729, 3)
        self.valid_mask = data['valid_mask']      # (N, 729)
        self.num_3d_frames = data['num_frames']
        
        print(f"[Llava3D] Loaded 3D cache: {self.num_3d_frames} frames, "
              f"valid ratio {self.valid_mask.float().mean():.1%}")
    
    def _get_frame_indices(self, video_path: str, max_frames: int) -> list:
        """获取视频采样帧索引"""
        import numpy as np
        from decord import VideoReader, cpu
        
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        return indices.tolist()
    
    def _load_3d_for_video(self, video_path: str) -> tuple:
        """
        为当前视频加载 3D 数据
        
        Returns:
            coords: (T, 729, 3)
            mask: (T, 729, 1)
        """
        if not self.use_3d:
            return None, None
        
        # 计算采样帧索引
        indices = self._get_frame_indices(video_path, self.max_frames_num)
        
        # 映射到 3D 帧索引（假设 1:1 或按比例映射）
        if self.num_3d_frames == len(indices):
            frame_3d_indices = list(range(self.num_3d_frames))
        else:
            # 按比例映射
            ratios = [i / len(indices) for i in range(len(indices))]
            frame_3d_indices = [int(r * (self.num_3d_frames - 1)) for r in ratios]
        
        # 提取对应帧的 3D 数据
        coords = self.centers_3d[frame_3d_indices]      # (T, 729, 3)
        mask = self.valid_mask[frame_3d_indices]        # (T, 729)
        
        return coords, mask.unsqueeze(-1)  # (T, 729, 3), (T, 729, 1)
    
    def generate_until(self, requests):
        """
        重写生成函数，注入 3D 数据
        """
        if not self.use_3d:
            return super().generate_until(requests)
        
        # 预处理：为每个请求加载 3D 数据
        video_3d_cache = {}
        
        for req in requests:
            contexts, doc_to_target, doc_to_visual, doc_id, task, split = req.args
            visual = doc_to_visual(self.task_dict[task][split][doc_id])
            
            if visual and isinstance(visual[0], str):
                video_path = visual[0]
                if video_path not in video_3d_cache:
                    coords, mask = self._load_3d_for_video(video_path)
                    if coords is not None:
                        video_3d_cache[video_path] = (coords, mask)
        
        # 包装 prepare_inputs 来注入 3D 数据
        original_prepare = self.model.prepare_inputs_labels_for_multimodal
        
        def wrapped_prepare(input_ids, position_ids, attention_mask, 
                           past_key_values, labels, images, **kwargs):
            # 注入 3D 数据（简化处理：使用第一个视频的 3D 数据）
            if video_3d_cache and images is not None:
                first_video = list(video_3d_cache.keys())[0]
                coords, mask = video_3d_cache[first_video]
                
                # 调整 batch 维度
                B = images.shape[0] if hasattr(images, 'shape') else 1
                if B > 1 and coords.shape[0] == 1:
                    coords = coords.expand(B, -1, -1)
                    mask = mask.expand(B, -1, -1)
                
                # 展平帧维度 (B, T, ...) -> (B*T, ...)
                coords_flat = coords.view(-1, coords.shape[-2], coords.shape[-1])
                mask_flat = mask.view(-1, mask.shape[-2], mask.shape[-1])
                
                self.adapter.set_3d_data(coords_flat, mask_flat)
            
            # 调用原始方法
            result = original_prepare(input_ids, position_ids, attention_mask,
                                     past_key_values, labels, images, **kwargs)
            
            # 清理
            self.adapter.clear_3d_data()
            
            return result
        
        # 替换方法
        self.model.prepare_inputs_labels_for_multimodal = wrapped_prepare
        
        try:
            # 调用父类生成
            results = super().generate_until(requests)
        finally:
            # 恢复原始方法
            self.model.prepare_inputs_labels_for_multimodal = original_prepare
        
        return results
