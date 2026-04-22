"""
残差 3D 位置编码 (Residual 3D Injection)
=========================================
通过简单的 MLP 将 3D 坐标映射到 vision feature 维度，以残差方式相加。

优点：
- 极少的可训练参数 (~100K)
- 无 cross-attention，显存友好
- 实现简单，收敛稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual3DInjection(nn.Module):
    """
    残差 3D 注入模块
    
    将 3D 坐标 (N, 3) 通过 MLP 映射到 feature 维度，
    以残差方式加到 vision features 上。
    """
    
    def __init__(
        self,
        feature_dim: int = 1152,  # SigLIP 输出维度
        hidden_dim: int = 256,    # MLP 隐藏层维度
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 3D 坐标投影 MLP: 3 -> hidden_dim -> feature_dim
        self.coord_proj = nn.Sequential(
            # 第一层：3D 坐标 -> 隐藏层
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # 第二层：隐藏层 -> feature_dim
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # 第三层：投影到最终维度
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        
        # 可学习的残差缩放因子
        self.residual_scale = nn.Parameter(torch.zeros(1))
        
        # 初始化
        self._init_weights()
        
        # 打印参数量
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Residual3D] Initialized with {num_params:,} trainable parameters")
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        image_features: torch.Tensor,  # (B, N, D)
        coords_3d: torch.Tensor,        # (B, N, 3)
        valid_mask: torch.Tensor = None,  # (B, N, 1) 可选
    ):
        """
        Args:
            image_features: 视觉特征 (B, N, D)
            coords_3d: 3D 坐标 (B, N, 3)
            valid_mask: 有效点掩码 (B, N, 1)
        
        Returns:
            enhanced_features: 增强后的特征 (B, N, D)
            info: 额外信息（用于调试）
        """
        B, N, D = image_features.shape
        assert D == self.feature_dim, f"Feature dim mismatch: {D} vs {self.feature_dim}"
        
        # 保存原始 dtype
        orig_dtype = image_features.dtype
        
        # 在 float32 下进行计算（更稳定）
        coords_float = coords_3d.float()
        
        # 投影 3D 坐标到 feature 空间
        coord_features = self.coord_proj(coords_float)  # (B, N, D)
        
        # 应用有效掩码（如果提供）
        if valid_mask is not None:
            # 确保 mask 维度正确
            if valid_mask.dim() == 2:
                valid_mask = valid_mask.unsqueeze(-1)  # (B, N) -> (B, N, 1)
            coord_features = coord_features * valid_mask.float()
        
        # 残差相加：缩放后的 3D 特征 + 原始视觉特征
        # 使用 sigmoid 限制缩放范围在 [0, 1]，初始值接近 0（逐渐学习）
        scale = torch.sigmoid(self.residual_scale)
        enhanced_features = image_features.float() + scale * coord_features
        
        # 转换回原始 dtype
        enhanced_features = enhanced_features.to(orig_dtype)
        
        info = {
            'residual_scale': scale.item(),
            'coord_feature_norm': coord_features.norm(dim=-1).mean().item(),
        }
        
        return enhanced_features, info


class Residual3DInjectionV2(nn.Module):
    """
    残差 3D 注入 V2 版本
    
    增加 per-frame 的时序聚合，适合视频场景。
    对每帧分别处理，然后聚合。
    """
    
    def __init__(
        self,
        feature_dim: int = 1152,
        hidden_dim: int = 256,
        num_frames: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        
        # 空间编码（每帧内部）
        self.spatial_proj = Residual3DInjection(feature_dim, hidden_dim, dropout)
        
        # 可学习的时序权重（用于加权多帧）
        self.temporal_weights = nn.Parameter(torch.ones(num_frames) / num_frames)
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Residual3D-V2] Initialized with {num_params:,} trainable parameters")
    
    def forward(
        self,
        image_features: torch.Tensor,  # (B*T, N, D) 或 (B, T*N, D)
        coords_3d: torch.Tensor,        # (B, T, N, 3)
        valid_mask: torch.Tensor = None,  # (B, T, N, 1)
        return_temporal: bool = False,
    ):
        """
        Args:
            image_features: 可以是展平形式 (B, T*N, D) 或分离形式 (B*T, N, D)
            coords_3d: 3D 坐标 (B, T, N, 3)
            valid_mask: 有效点掩码 (B, T, N, 1)
        """
        B, T, N, _ = coords_3d.shape
        
        # 如果 features 是展平的，需要 reshape
        if image_features.shape[1] == T * N:
            # (B, T*N, D) -> (B*T, N, D)
            image_features = image_features.view(B * T, N, -1)
        
        # reshape 3D 坐标和 mask
        coords_flat = coords_3d.view(B * T, N, 3)  # (B*T, N, 3)
        if valid_mask is not None:
            mask_flat = valid_mask.view(B * T, N, 1)  # (B*T, N, 1)
        else:
            mask_flat = None
        
        # 空间编码
        enhanced_flat, info = self.spatial_proj(image_features, coords_flat, mask_flat)
        
        # reshape 回 (B, T, N, D)
        enhanced = enhanced_flat.view(B, T, N, -1)
        
        if return_temporal:
            # 应用时序加权
            weights = F.softmax(self.temporal_weights[:T], dim=0)  # (T,)
            temporal_pooled = (enhanced * weights.view(1, T, 1, 1)).sum(dim=1)  # (B, N, D)
            return temporal_pooled, info
        
        # 展平返回 (B, T*N, D)
        return enhanced.view(B, T * N, -1), info
