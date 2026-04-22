"""
LLaVA-OneVision + 3D Position Encoding Adapter
===============================================
在 vision_tower 和 mm_projector 之间插入 3D 位置编码
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/home/qyk/map-anything")
from pipeline_3d_vlm.pos_encoder_3d import HybridPositionalEncoding3D


class LLaVA3DAdapter(nn.Module):
    """
    包装器：在 LLaVA-OneVision 的 vision encoder 中插入 3D 位置编码
    
    架构：
    images -> vision_tower -> 3D Pos Encoder -> mm_projector -> LLM
              (frozen)         (trainable)      (frozen)
    """
    
    def __init__(
        self,
        base_model,  # LLaVA-OneVision model
        freeze_mode: bool = True,
        coord_dim: int = 1152,
        **pos_encoder_kwargs
    ):
        super().__init__()
        self.base_model = base_model
        self.freeze_mode = freeze_mode
        
        # 获取 vision_tower 和 projector
        self.vision_tower = base_model.get_model().get_vision_tower()
        self.mm_projector = base_model.get_model().mm_projector
        
        # 初始化 3D 位置编码器
        self.pos_encoder = HybridPositionalEncoding3D(
            dim=coord_dim,
            **pos_encoder_kwargs
        )
        
        if freeze_mode:
            self._freeze_base_model()
            print("[LLaVA3DAdapter] Freeze mode: Only 3D encoder trainable")
        
        self._current_3d_data = None
    
    def _freeze_base_model(self):
        """冻结原有模型参数"""
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        for param in self.mm_projector.parameters():
            param.requires_grad = False
        for param in self.base_model.get_model().parameters():
            param.requires_grad = False
    
    def set_3d_data(self, coords_3d: torch.Tensor, valid_mask: torch.Tensor):
        """设置当前 batch 的 3D 数据"""
        self._current_3d_data = {
            'coords': coords_3d,
            'mask': valid_mask
        }
    
    def clear_3d_data(self):
        self._current_3d_data = None
    
    def encode_images_with_3d(self, images):
        """替换 encode_images，加入 3D 位置编码"""
        image_features = self.vision_tower(images)
        if isinstance(image_features, tuple):
            image_features = image_features[0]
        
        if self._current_3d_data is not None:
            coords = self._current_3d_data['coords'].to(images.device)
            mask = self._current_3d_data['mask'].to(images.device)
            image_features, _ = self.pos_encoder(image_features, coords, mask)
        
        image_features = self.mm_projector(image_features)
        return image_features
    
    def get_trainable_params(self):
        return [p for p in self.pos_encoder.parameters() if p.requires_grad]
