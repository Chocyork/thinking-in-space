"""
配置文件：LLaVA-OneVision-3D 训练/推理配置
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LLaVA3DConfig:
    """LLaVA-OneVision-3D 配置"""
    
    # ==================== 模型配置 ====================
    model_name: str = "llava-onevision-qwen2-7b-ov"
    siglip_dim: int = 1152  # SigLIP 输出维度
    
    # ==================== 3D 编码器配置 ====================
    pos_encoder_dim: int = 1152
    pos_encoder_hidden: int = 512
    pos_encoder_layers: int = 2
    pos_encoder_dropout: float = 0.1
    
    # ==================== 训练配置 ====================
    # 两种模式：
    # 1. freeze_base=True: 只训练 3D 编码器（推荐初期使用）
    # 2. freeze_base=False: 端到端微调（需要更多数据）
    freeze_base: bool = True
    
    # 学习率配置
    lr_3d_encoder: float = 1e-4      # 3D 编码器学习率
    lr_fine_tune: float = 1e-5       # 基础模型学习率（freeze_base=False 时使用）
    weight_decay: float = 0.01
    
    # 训练参数
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    
    # ==================== 数据配置 ====================
    point_cloud_dir: str = "/home/qyk/map-anything/3d_cache"
    max_frames: int = 8
    
    # ==================== 推理配置 ====================
    use_3d: bool = True
    
    def to_dict(self):
        return {
            'model_name': self.model_name,
            'siglip_dim': self.siglip_dim,
            'pos_encoder_kwargs': {
                'dim': self.pos_encoder_dim,
                'coord_hidden_dim': self.pos_encoder_hidden,
                'num_fusion_layers': self.pos_encoder_layers,
                'dropout': self.pos_encoder_dropout,
            },
            'freeze_base': self.freeze_base,
            'learning_rate': self.lr_3d_encoder if self.freeze_base else self.lr_fine_tune,
        }


# 预定义配置
def get_freeze_config():
    """冻结模式配置（推荐）"""
    return LLaVA3DConfig(
        freeze_base=True,
        lr_3d_encoder=1e-4,
        num_epochs=3,
    )


def get_finetune_config():
    """微调模式配置（需要更多数据）"""
    return LLaVA3DConfig(
        freeze_base=False,
        lr_fine_tune=1e-5,
        num_epochs=5,
    )
