#!/usr/bin/env python3
"""
VSI-Bench 3D VLM 训练脚本 - 残差注入版本
==========================================
使用 Residual3DInjection 替代 HybridPositionalEncoding3D
- 参数量减少 40 倍（~100K vs 4.5M）
- 无 cross-attention，显存友好
- 支持单卡/多卡(DDP)训练

使用方法:
    bash train_residual.sh          # 单卡
    bash train_residual_4gpu.sh     # 4卡
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
from PIL import Image

# 添加 LLaVA 路径
sys.path.insert(0, "/home/qyk/thinking-in-space/LLaVA-NeXT")
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images

# 添加项目路径
sys.path.insert(0, "/home/qyk/map-anything")
sys.path.insert(0, "/home/qyk/thinking-in-space")

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from lmms_eval.models.pos_encoder_3d_residual import Residual3DInjection

# DeepSpeed
try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False


class VSIBench3DDataset(Dataset):
    """VSI-Bench 数据集 - 3D 版本"""
    
    def __init__(
        self,
        data_root="/mnt/data/qyk/nyu-visionx/VSI-Bench",
        video_root="/mnt/data/qyk/43d",
        cache_dir="/mnt/data/qyk/43d3dpt",
        split="train",
        max_frames=32,
    ):
        self.data_root = Path(data_root)
        self.video_root = Path(video_root) if video_root else self.data_root
        self.cache_dir = Path(cache_dir)
        self.max_frames = max_frames
        
        # 加载 annotation
        self.annotations = self._load_annotations()
        
        # 过滤有 3D 缓存的样本
        self.valid_samples = self._filter_valid_samples()
        
        print(f"[Dataset] Loaded {len(self.valid_samples)} valid samples from {split}")
    
    def _load_annotations(self):
        """加载 VSI-Bench 标注"""
        import pandas as pd
        
        ann_files = [
            self.data_root / "test_debiased.parquet",
            self.data_root / "test.jsonl",
        ]
        
        for ann_file in ann_files:
            if ann_file.exists():
                if ann_file.suffix == '.parquet':
                    df = pd.read_parquet(ann_file)
                    return df.to_dict('records')
                else:
                    with open(ann_file) as f:
                        return [json.loads(line) for line in f]
        
        raise FileNotFoundError(f"No annotation file found in {self.data_root}")
    
    def _filter_valid_samples(self):
        """过滤有 3D 缓存的有效样本"""
        valid = []
        
        for ann in self.annotations:
            dataset = ann.get('dataset', 'unknown')
            scene = ann.get('scene_name', '')
            
            cache_names = [
                f"{scene}_3d.pt",
                f"{dataset}_{scene}_3d.pt",
            ]
            
            found = False
            for name in cache_names:
                if (self.cache_dir / name).exists():
                    ann['cache_file'] = name
                    # 构造视频路径（指向帧目录）
                    video_path = self.video_root / dataset / scene
                    if video_path.exists():
                        ann['video_path'] = str(video_path)
                        valid.append(ann)
                        found = True
                        break
        
        return valid
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        ann = self.valid_samples[idx]
        
        # 加载 3D 缓存
        cache_path = self.cache_dir / ann['cache_file']
        cache_data = torch.load(cache_path, map_location='cpu')
        
        # 根据 max_frames 采样 3D 数据
        total_frames = cache_data['centers_3d'].shape[0]
        if total_frames > self.max_frames:
            indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
            coords_3d = cache_data['centers_3d'][indices]
            valid_mask = cache_data['valid_mask'][indices]
        else:
            coords_3d = cache_data['centers_3d']
            valid_mask = cache_data['valid_mask']
        
        return {
            'question': ann['question'],
            'answer': str(ann.get('ground_truth', ann.get('answer', ''))),
            'question_type': ann.get('question_type', 'unknown'),
            'cache_file': str(cache_path),
            'video_path': ann.get('video_path', ''),
            'coords_3d': coords_3d,  # (T, N, 3)
            'valid_mask': valid_mask,  # (T, N, 1)
            'dataset': ann.get('dataset', 'unknown'),
        }


def collate_fn(batch):
    """自定义 collate 函数"""
    return batch


class SimpleTrainer3D:
    """简化的 3D VLM 训练器（支持DDP）"""
    
    def __init__(self, args):
        self.args = args
        
        # 初始化分布式环境
        self._init_distributed()
        
        # 创建输出目录（只在主进程）
        if self.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # DeepSpeed 初始化（如果使用）
        self.ds_engine = None
        if self.args.deepspeed:
            if not HAS_DEEPSPEED:
                raise ImportError("DeepSpeed not installed")
            # 先初始化模型，然后 DeepSpeed 会接管
            self._init_model()
            self._init_dataloader()
            self._init_deepspeed()
        else:
            self._init_model()
            self._init_dataloader()
            self._init_optimizer()
        
        # 训练状态
        self.global_step = 0
        self.best_loss = float('inf')
    
    def _init_distributed(self):
        """初始化分布式训练环境"""
        # 检查是否处于分布式环境
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.distributed = True
            
            # 初始化进程组
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            
            print(f"[DDP] Rank {self.rank}/{self.world_size}, Local rank {self.local_rank}")
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.distributed = False
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @property
    def is_main_process(self):
        return self.rank == 0
    
    def _init_model(self):
        """初始化模型"""
        print("[Trainer] Initializing model...")
        
        # 加载 LLaVA-OneVision
        from lmms_eval.models.llava_onevision import Llava_OneVision
        
        self.model = Llava_OneVision(
            pretrained="/mnt/data/qyk/lmms-lab/llava-onevision-qwen2-0.5b-ov/",
        )
        
        # 冻结 LLM 和 Vision Tower
        for param in self.model.model.parameters():
            param.requires_grad = False
        
        # 解冻 Projector（可选）
        if self.args.train_projector:
            for param in self.model.model.get_model().mm_projector.parameters():
                param.requires_grad = True
                param.data = param.data.float()  # 确保 FP32
            print("[Trainer] Projector unfrozen")
        
        # 添加残差 3D 注入模块（保持 FP32 以避免 FP16 梯度问题）
        self.pos_encoder = Residual3DInjection(
            feature_dim=1152,
            hidden_dim=self.args.hidden_dim,
            dropout=0.1
        ).to(self.device).float()  # 确保 FP32
        
        # Hook encode_images
        self._original_encode_images = self.model.model.encode_images
        self.model.model.encode_images = self._encode_images_with_3d
        
        # 启用 gradient checkpointing
        if hasattr(self.model.model, 'gradient_checkpointing_enable'):
            self.model.model.gradient_checkpointing_enable()
            print("[Trainer] Gradient checkpointing enabled")
        
        # 设置训练模式
        self.pos_encoder.train()
        self.model.model.eval()
        
        # DDP包装（如果是分布式）
        if self.distributed:
            self.pos_encoder = DDP(
                self.pos_encoder, 
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
            if self.args.train_projector:
                self.model.model.get_model().mm_projector = DDP(
                    self.model.model.get_model().mm_projector,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank
                )
        
        # 打印参数量（只在主进程）
        if self.is_main_process:
            trainable_params = sum(p.numel() for p in self.pos_encoder.parameters() if p.requires_grad)
            if self.args.train_projector:
                trainable_params += sum(p.numel() for p in self.model.model.get_model().mm_projector.parameters() if p.requires_grad)
            print(f"[Trainer] Trainable params: {trainable_params:,}")
    
    def _encode_images_with_3d(self, images):
        """Hook：加入 3D 注入"""
        # 1. Vision Tower
        image_features = self.model.model.get_vision_tower()(images)
        if isinstance(image_features, tuple):
            image_features = image_features[0]
        
        # 2. 残差 3D 注入（仅在数据存在时）
        if hasattr(self, '_current_3d_coords') and self._current_3d_coords is not None:
            coords = self._current_3d_coords.to(images.device)
            mask = self._current_3d_mask.to(images.device)
            
            # 展平时序 (B*T, N, D)
            B, N, D = image_features.shape
            T = coords.shape[0]
            
            # 如果时序维度不匹配，需要 reshape
            if B == T:
                # 每帧一个样本
                coords = coords  # (T, N, 3)
                mask = mask.squeeze(-1) if mask.dim() == 3 else mask  # (T, N)
            else:
                # 多帧展平
                coords = coords.view(-1, N, 3)  # (B*T, N, 3)
                mask = mask.view(-1, N)  # (B*T, N)
            
            # 应用残差注入
            image_features, _ = self.pos_encoder(
                image_features, coords, mask
            )
        
        # 3. Projector
        image_features = self.model.model.get_model().mm_projector(image_features)
        return image_features
    
    def _init_dataloader(self):
        """初始化数据加载器（支持DDP）"""
        if self.is_main_process:
            print("[Trainer] Loading dataset...")
        
        dataset = VSIBench3DDataset(
            data_root=self.args.data_root,
            video_root=self.args.video_root,
            cache_dir=self.args.cache_dir,
            split="train",
            max_frames=self.args.max_frames,
        )
        
        # 分布式采样器
        if self.distributed:
            sampler = DistributedSampler(
                dataset, 
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
        )
    
    def _init_optimizer(self):
        """初始化优化器"""
        # 收集可训练参数
        trainable_params = list(self.pos_encoder.parameters())
        if self.args.train_projector:
            trainable_params += list(self.model.model.get_model().mm_projector.parameters())
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        
        total_steps = len(self.dataloader) * self.args.num_epochs
        warmup_steps = int(total_steps * self.args.warmup_ratio)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        print(f"[Trainer] Optimizer: AdamW, LR: {self.args.learning_rate}")
    
    def _init_deepspeed(self):
        """初始化 DeepSpeed 引擎"""
        print("[Trainer] Initializing DeepSpeed with ZeRO-2...")
        
        # 从文件加载配置
        with open(self.args.deepspeed, 'r') as f:
            ds_config = json.load(f)
        
        # 更新 batch size
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        ds_config["train_batch_size"] = self.args.batch_size * world_size
        ds_config["train_micro_batch_size_per_gpu"] = self.args.batch_size
        
        # 简单初始化 - 让 DeepSpeed 管理所有参数
        self.ds_engine = deepspeed.initialize(
            model=self.model.model,
            config=ds_config
        )[0]
        
        print(f"[Trainer] DeepSpeed initialized with ZeRO-2")
        print(f"[Trainer] World size: {world_size}, Local rank: {self.ds_engine.local_rank}")
    
    def _load_video_frames(self, video_path):
        """加载视频帧"""
        import os
        
        if os.path.isdir(video_path):
            # 帧图片目录
            frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            total_frames = len(frame_files)
            
            # 均匀采样
            indices = np.linspace(0, total_frames - 1, self.args.max_frames, dtype=int)
            
            frames = []
            for idx in indices:
                frame_path = os.path.join(video_path, frame_files[idx])
                frame = Image.open(frame_path).convert('RGB')
                frames.append(frame)
            return frames
        else:
            # 视频文件
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            indices = np.linspace(0, total_frames - 1, self.args.max_frames, dtype=int)
            frames = vr.get_batch(indices.tolist()).asnumpy()
            return [Image.fromarray(f) for f in frames]
    
    def _process_frames(self, frames):
        """预处理帧"""
        image_processor = self.model._image_processor
        pixel_values = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        return pixel_values.half().to(self.device) if self.args.fp16 else pixel_values.to(self.device)
    
    def _build_conversation(self, question, answer, num_frames):
        """构建对话"""
        import copy
        
        prompts_input = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], prompts_input)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(
            prompt, self.model._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).squeeze(0).to(self.device)
        
        conv.messages[-1][1] = answer
        full_prompt = conv.get_prompt()
        full_input_ids = tokenizer_image_token(
            full_prompt, self.model._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).squeeze(0).to(self.device)
        
        labels = full_input_ids.clone()
        labels[:input_ids.shape[0]] = -100
        
        return full_input_ids.unsqueeze(0), labels.unsqueeze(0)
    
    def train(self):
        """训练循环（支持DDP）"""
        if self.is_main_process:
            print(f"\n{'='*60}")
            print(f"Starting Training (Residual 3D Injection)")
            print(f"World Size: {self.world_size}")
            print(f"{'='*60}\n")
        
        for epoch in range(self.args.num_epochs):
            # 设置分布式采样器的epoch
            if self.distributed and hasattr(self.dataloader, 'sampler') and isinstance(self.dataloader.sampler, DistributedSampler):
                self.dataloader.sampler.set_epoch(epoch)
            
            self.pos_encoder.train()
            
            # 只在主进程显示进度条
            dataloader_iter = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}") if self.is_main_process else self.dataloader
            
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader_iter:
                loss = self._train_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                lr = self.scheduler.get_last_lr()[0]
                
                # 更新进度条（只在主进程）
                if self.is_main_process:
                    dataloader_iter.set_postfix({
                        'loss': f'{loss:.4f}',
                        'avg_loss': f'{epoch_loss/num_batches:.4f}',
                        'lr': f'{lr:.2e}',
                    })
                
                # 保存 checkpoint（只在主进程）
                if self.global_step % self.args.save_steps == 0 and self.is_main_process:
                    self._save_checkpoint(f"step_{self.global_step}")
                
                self.global_step += 1
            
            # 同步 epoch_loss 到所有进程
            if self.distributed:
                epoch_loss_tensor = torch.tensor(epoch_loss / num_batches, device=self.device)
                dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.AVG)
                avg_epoch_loss = epoch_loss_tensor.item()
            else:
                avg_epoch_loss = epoch_loss / num_batches
            
            if self.is_main_process:
                print(f"\n[Epoch {epoch+1}] Avg Loss: {avg_epoch_loss:.4f}")
            
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                if self.is_main_process:
                    self._save_checkpoint("best")
                    print(f"  ✓ New best model saved")
            
            if self.is_main_process:
                self._save_checkpoint(f"epoch_{epoch+1}")
        
        if self.is_main_process:
            print(f"\n{'='*60}")
            print("Training Complete!")
            print(f"Best Loss: {self.best_loss:.4f}")
            print(f"{'='*60}")
    
    def _train_step(self, batch):
        """单步训练"""
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        valid_samples = 0
        
        for sample in batch:
            try:
                # 1. 加载视频帧
                video_path = sample['video_path']
                if not video_path or not os.path.exists(video_path):
                    continue
                
                frames = self._load_video_frames(video_path)
                image_tensor = self._process_frames(frames)
                
                # 2. 准备 3D 数据
                coords = sample['coords_3d'].to(self.device)
                mask = sample['valid_mask'].to(self.device)
                
                # 设置当前 3D 数据
                self._current_3d_coords = coords
                self._current_3d_mask = mask.squeeze(-1) if mask.dim() == 3 else mask
                
                # 3. 构造对话
                question = sample['question']
                answer = sample['answer']
                input_ids, labels = self._build_conversation(question, answer, len(frames))
                
                # 4. 前向传播
                with autocast(enabled=self.args.fp16):
                    outputs = self.model.model(
                        input_ids=input_ids,
                        labels=labels,
                        images=[image_tensor],
                        modalities=["video"],
                    )
                    loss = outputs.loss
                
                # 5. 反向传播
                if self.ds_engine is not None:
                    self.ds_engine.backward(loss)
                else:
                    loss.backward()
                
                total_loss += loss.item()
                valid_samples += 1
                
                # 清理显存（非 DeepSpeed 模式）
                if self.ds_engine is None:
                    del loss, outputs
                    torch.cuda.empty_cache()
                
                # 清理
                delattr(self, '_current_3d_coords')
                delattr(self, '_current_3d_mask')
                
            except Exception as e:
                print(f"[Warning] Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 更新
        if valid_samples > 0:
            if self.ds_engine is not None:
                # DeepSpeed 模式
                self.ds_engine.step()
            else:
                # 普通模式：梯度裁剪 + 优化器步进
                torch.nn.utils.clip_grad_norm_(
                    list(self.pos_encoder.parameters()) + 
                    (list(self.model.model.get_model().mm_projector.parameters()) if self.args.train_projector else []),
                    max_norm=1.0
                )
                self.optimizer.step()
                self.scheduler.step()
            
            return total_loss / valid_samples
        return 0.0
    
    def _save_checkpoint(self, name):
        """保存检查点（只在主进程）"""
        if not self.is_main_process:
            return
        
        checkpoint_dir = Path(self.args.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        state_dict = {
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'args': vars(self.args),
        }
        
        # 获取模型状态（处理DDP的module.前缀）
        pos_encoder = self.pos_encoder.module if self.distributed else self.pos_encoder
        
        # 保存 pos_encoder
        pos_encoder_state = {}
        for k, v in pos_encoder.state_dict().items():
            pos_encoder_state[f'pos_encoder.{k}'] = v.cpu()
        state_dict.update(pos_encoder_state)
        
        # 保存 projector（如果训练了）
        if self.args.train_projector:
            projector = self.model.model.get_model().mm_projector
            if self.distributed:
                projector = projector.module
            projector_state = {}
            for k, v in projector.state_dict().items():
                projector_state[f'projector.{k}'] = v.cpu()
            state_dict.update(projector_state)
        
        torch.save(state_dict, checkpoint_dir / "trainable_weights.pt")
        print(f"  ✓ Saved checkpoint to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser()
    
    # 数据
    parser.add_argument("--data_root", type=str, default="/mnt/data/qyk/nyu-visionx/VSI-Bench")
    parser.add_argument("--video_root", type=str, default="/mnt/data/qyk/43d")
    parser.add_argument("--cache_dir", type=str, default="/mnt/data/qyk/43d3dpt")
    
    # 训练
    parser.add_argument("--output_dir", type=str, default="./checkpoints/3d_residual")
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)  # 残差注入可以用更大学习率
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    
    # 模型
    parser.add_argument("--train_projector", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DeepSpeed")
    
    args = parser.parse_args()
    
    # 训练
    trainer = SimpleTrainer3D(args)
    try:
        trainer.train()
    finally:
        # 清理分布式环境
        if trainer.distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
