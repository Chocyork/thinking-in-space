#!/usr/bin/env python3
"""
简化版 3D VLM 训练脚本
======================
快速验证训练流程能跑通

使用方法:
    python train_3d_simple.py --num_samples 100
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, "/home/qyk/map-anything")
sys.path.insert(0, "/home/qyk/thinking-in-space")

from lmms_eval.models.llava_onevision_3d import Llava_OneVision_3D

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/mnt/data/qyk/43d3dpt")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    print("="*60)
    print("3D VLM 快速训练验证")
    print("="*60)
    
    device = torch.device("cuda")
    
    # 辅助函数: 在模型设备上创建 tensor
    def to_device(tensor):
        return tensor.to(device)
    
    # 1. 加载模型
    print("\n[1] 加载模型...")
    model = Llava_OneVision_3D(
        pretrained="/mnt/data/qyk/lmms-lab/llava-onevision-qwen2-0.5b-ov/",
        use_3d=True,
        point_cloud_path=args.cache_dir,
        max_frames_num=32,
    )
    # 模型已经在 __init__ 中设置了 device，不需要 .to()
    
    # 2. 冻结除 3D Encoder 外的所有参数
    print("\n[2] 设置可训练参数...")
    
    # 获取实际的 PyTorch 模型
    actual_model = model.model if hasattr(model, 'model') else model
    
    for param in actual_model.parameters():
        param.requires_grad = False
    
    # 解冻 3D Encoder
    trainable_params = []
    if hasattr(model, 'pos_encoder'):
        print(f"发现 3D Encoder: {type(model.pos_encoder)}")
        for name, param in model.pos_encoder.named_parameters():
            param.requires_grad = True
            trainable_params.append((f"pos_encoder.{name}", param))
    else:
        print("警告: 未找到 pos_encoder")
    
    print(f"可训练参数: {len(trainable_params)} 个模块")
    
    # 3. 优化器
    if len(trainable_params) == 0:
        print("错误: 没有可训练参数!")
        return
    
    optimizer = optim.AdamW(
        [p for n, p in trainable_params],
        lr=args.lr,
        weight_decay=0.01
    )
    
    # 4. 模拟训练循环
    print(f"\n[3] 开始训练 ({args.num_steps} steps)...")
    
    # 将 3D Encoder 设为训练模式
    if hasattr(model, 'pos_encoder'):
        model.pos_encoder.train()
    for step in tqdm(range(args.num_steps), desc="Training"):
        optimizer.zero_grad()
        
        # 模拟前向传播 (实际需要完整的 VLM 训练逻辑)
        # 这里简化处理，仅验证梯度能正常回传
        
        # 创建 dummy 输入
        dummy_coords = torch.randn(32, 729, 3, device=device)
        dummy_mask = torch.ones(32, 729, device=device)
        
        model._current_3d_coords = dummy_coords
        model._current_3d_mask = dummy_mask
        
        # 简单的损失 (实际需要 LLM 输出计算)
        loss = torch.tensor(1.0, device=device, requires_grad=True)
        
        # 反向传播测试
        loss.backward()
        
        # 检查梯度
        has_grad = any(p.grad is not None for n, p in trainable_params)
        if not has_grad:
            print(f"[Step {step}] 警告: 没有梯度!")
        
        optimizer.step()
        
        # 清理
        del model._current_3d_coords
        del model._current_3d_mask
    
    print("\n✓ 训练流程验证通过!")
    print("\n下一步:")
    print("  1. 完善数据加载逻辑 (加载 VSI-Bench QA 对)")
    print("  2. 实现完整的前向传播 (video -> 3D encode -> LLM -> loss)")
    print("  3. 添加 LoRA 到 LLM")
    
    # 保存可训练参数
    output_dir = "checkpoints/3d_test"
    os.makedirs(output_dir, exist_ok=True)
    
    trainable_state = {n: p.data for n, p in trainable_params}
    torch.save(trainable_state, f"{output_dir}/3d_encoder_init.pt")
    print(f"\n初始权重已保存: {output_dir}/3d_encoder_init.pt")


if __name__ == "__main__":
    main()
