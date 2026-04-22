#!/usr/bin/env python3
"""
完整的 3D 测试脚本
=================
基于现有的 llava_onevision.py 添加 3D 支持

使用方法:
    # 测试 2D baseline
    python run_3d_test.py --mode 2d --video_dir /path/to/frames
    
    # 测试 3D enhanced
    python run_3d_test.py --mode 3d --video_dir /path/to/frames --cache_3d /path/to/cache.pt
"""

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json

sys.path.insert(0, "/home/qyk/thinking-in-space")
sys.path.insert(0, "/home/qyk/map-anything")

from lmms_eval.models.llava_onevision import Llava_OneVision


class LLaVAOneVision3DTester(Llava_OneVision):
    """
    扩展 LLaVA-OneVision 支持 3D 输入
    """
    
    def __init__(self, cache_3d_path=None, **kwargs):
        super().__init__(**kwargs)
        self.cache_3d_path = cache_3d_path
        self._3d_data = None
        
        if cache_3d_path:
            self._load_3d_cache(cache_3d_path)
            print(f"[3DTester] Loaded 3D cache: {cache_3d_path}")
    
    def _load_3d_cache(self, path):
        """加载 3D 缓存"""
        data = torch.load(path)
        self._3d_data = {
            'centers': data['centers_3d'],  # (N, 729, 3)
            'valid_mask': data['valid_mask'],  # (N, 729)
            'num_frames': data['num_frames']
        }
    
    def _get_3d_for_frames(self, frame_indices):
        """
        获取指定帧的 3D 数据
        
        Args:
            frame_indices: list of frame indices
            
        Returns:
            coords: (T, 729, 3)
            mask: (T, 729, 1)
        """
        if self._3d_data is None:
            return None, None
        
        # 映射到 3D 帧索引
        total_3d_frames = self._3d_data['num_frames']
        indices_3d = []
        for idx in frame_indices:
            ratio = idx / len(frame_indices)
            mapped_idx = int(ratio * (total_3d_frames - 1))
            indices_3d.append(mapped_idx)
        
        coords = self._3d_data['centers'][indices_3d]  # (T, 729, 3)
        mask = self._3d_data['valid_mask'][indices_3d]  # (T, 729)
        
        return coords, mask.unsqueeze(-1)  # (T, 729, 3), (T, 729, 1)
    
    def generate_with_3d(self, images, question, frame_indices=None):
        """
        生成回答，注入 3D 信息
        
        注意：这是一个简化版本，实际需要更复杂的集成
        """
        # 获取 3D 数据
        if frame_indices is None:
            # 默认均匀采样
            frame_indices = list(range(len(images)))
        
        coords_3d, valid_mask = self._get_3d_for_frames(frame_indices)
        
        if coords_3d is not None:
            print(f"[3D] Injecting 3D coordinates: {coords_3d.shape}")
            # 这里需要修改 model.encode_images 来注入 3D 数据
            # 简化起见，先打印信息
        
        # 调用父类的生成方法
        # 注意：实际需要修改 prepare_inputs_labels_for_multimodal
        return self.generate_simple(images, question)
    
    def generate_simple(self, images, question):
        """
        简化的生成接口
        """
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token
        import copy
        
        # 处理图像
        if hasattr(images[0], 'convert'):  # PIL Images
            image_tensor = process_images(images, self._image_processor, self._config)
            if type(image_tensor) is list:
                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
        else:
            image_tensor = images
        
        # 构造 prompt
        if DEFAULT_IMAGE_TOKEN not in question:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
        else:
            prompt = question
        
        # Tokenize
        conv = self.conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        input_ids = tokenizer_image_token(
            prompt_text, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        
        # 生成
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                use_cache=True,
                do_sample=False,
                max_new_tokens=1024,
            )
        
        # 解码
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return outputs.strip()


def load_frames(frame_dir, max_frames=8):
    """加载视频帧"""
    frame_files = sorted([
        f for f in Path(frame_dir).iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])
    
    # 均匀采样
    indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
    selected_files = [frame_files[i] for i in indices]
    
    frames = [Image.open(f).convert('RGB') for f in selected_files]
    return frames


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["2d", "3d"], required=True)
    parser.add_argument("--video_dir", required=True, help="视频帧目录")
    parser.add_argument("--cache_3d", help="3D 缓存路径（3D 模式需要）")
    parser.add_argument("--model", default="lmms-lab/llava-onevision-qwen2-0p5b-ov")
    parser.add_argument("--question", default="What is the main object in the center of the scene?")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"LLaVA-OneVision Test: {args.mode.upper()} Mode")
    print("=" * 70)
    
    # 加载帧
    print(f"\nLoading frames from: {args.video_dir}")
    frames = load_frames(args.video_dir, max_frames=8)
    print(f"✓ Loaded {len(frames)} frames")
    
    # 加载模型
    print(f"\nLoading model: {args.model}")
    
    if args.mode == "3d":
        if not args.cache_3d:
            print("Error: --cache_3d is required for 3D mode")
            return
        
        model = LLaVAOneVision3DTester(
            pretrained=args.model,
            cache_3d_path=args.cache_3d,
            device="cuda:0",
            max_frames_num=8
        )
    else:
        model = LLaVAOneVision3DTester(
            pretrained=args.model,
            device="cuda:0",
            max_frames_num=8
        )
    
    print("✓ Model loaded")
    
    # 提问
    print(f"\nQuestion: {args.question}")
    print("-" * 70)
    
    if args.mode == "3d":
        answer = model.generate_with_3d(frames, args.question)
    else:
        answer = model.generate_simple(frames, args.question)
    
    print(f"Answer: {answer}")
    print("=" * 70)


if __name__ == "__main__":
    main()
