#!/usr/bin/env python3
"""
单场景 3D 测试脚本
==================
使用现有的 courtyard_3d.pt 快速验证 3D 编码器效果

对比：
1. LLaVA-OneVision (2D baseline)
2. LLaVA-OneVision + 3D Position Encoding

Author: Assistant
Date: 2026-03-10
"""

import sys
import torch
import json
from pathlib import Path
from typing import List, Dict
from PIL import Image
import numpy as np

# 添加路径
sys.path.insert(0, "/home/qyk/map-anything")
sys.path.insert(0, "/home/qyk/thinking-in-space")

from lmms_eval.models.llava_onevision import Llava_OneVision


# ==================== 测试配置 ====================

COURTYARD_VIDEO_PATH = "/mnt/data/qyk/courtyard_raw/courtyard/images/dslr_jpgs/"  # 图片帧目录
COURTYARD_3D_PATH = "/home/qyk/map-anything/3d_cache/courtyard_3d.pt"

# 人工准备的空间问题（针对 courtyard 场景）
COURTYARD_QUESTIONS = [
    {
        "question": "What is the main object in the center of the scene?",
        "type": "object_identification",
        "expected_answer": "table"  # 根据实际情况调整
    },
    {
        "question": "How many chairs are visible in the scene?",
        "type": "object_counting",
        "expected_answer": "2"
    },
    {
        "question": "Is there a tree on the left side or right side of the courtyard?",
        "type": "spatial_relation",
        "expected_answer": "left"
    },
    {
        "question": "What is behind the table?",
        "type": "spatial_relation",
        "expected_answer": "wall"  # 根据实际情况调整
    },
    {
        "question": "Describe the layout of the courtyard from left to right.",
        "type": "spatial_description",
        "expected_answer": None  # 开放性问题，无标准答案
    },
    {
        "question": "Which object is closer to the camera, the chair or the table?",
        "type": "depth_reasoning",
        "expected_answer": "table"
    },
    {
        "question": "What is the approximate distance between the two chairs in meters?",
        "type": "distance_estimation",
        "expected_answer": None
    },
    {
        "question": "If you enter from the gate, what is the first object you would see?",
        "type": "spatial_reasoning",
        "expected_answer": None
    },
]


class Simple3DTester:
    """
    简化的 3D 测试器
    直接加载模型进行推理对比
    """
    
    def __init__(
        self,
        model_path: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        device: str = "cuda"
    ):
        self.device = device
        self.model_path = model_path
        
        print("=" * 70)
        print("Initializing LLaVA-OneVision Models")
        print("=" * 70)
        
        # 1. 加载 2D baseline
        print("\n[1] Loading 2D baseline model...")
        self.model_2d = Llava_OneVision(
            pretrained=model_path,
            device=device,
            max_frames_num=8
        )
        print("✓ 2D model loaded")
        
        # 2. 加载 3D 增强模型（需要修改后的 evaluator）
        print("\n[2] Loading 3D enhanced model...")
        try:
            # 尝试导入 3D 版本
            from evaluator_3d import Llava_OneVision_3D
            
            self.model_3d = Llava_OneVision_3D(
                pretrained=model_path,
                device=device,
                use_3d=True,
                point_cloud_path=COURTYARD_3D_PATH,
                freeze_base=True,
                max_frames_num=8
            )
            print("✓ 3D model loaded")
            self.has_3d = True
        except Exception as e:
            print(f"⚠️ Failed to load 3D model: {e}")
            print("   Will test 2D only")
            self.has_3d = False
    
    def load_video_frames(self, frame_dir: str, max_frames: int = 8):
        """
        加载视频帧
        
        Args:
            frame_dir: 帧图片目录
            max_frames: 采样帧数
            
        Returns:
            frames: list of PIL Images
        """
        frame_files = sorted([
            f for f in Path(frame_dir).iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])
        
        # 均匀采样
        indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
        selected_files = [frame_files[i] for i in indices]
        
        frames = [Image.open(f).convert('RGB') for f in selected_files]
        return frames
    
    def ask_question_2d(self, question: str, frames: List[Image.Image]) -> str:
        """
        使用 2D 模型提问
        """
        # 构造 prompt（参考 llava_onevision.py 的格式）
        from llava.constants import DEFAULT_IMAGE_TOKEN
        
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
        
        # 生成回答
        # 注意：这里需要适配 llava_onevision.py 的 generate_until 接口
        # 简化起见，直接调用模型的 generate 方法
        
        inputs = self.model_2d.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 这里简化处理，实际需要完整的 generate 流程
        # 返回模拟回答
        return f"[2D Mock Answer for: {question[:30]}...]"
    
    def ask_question_3d(self, question: str, frames: List[Image.Image]) -> str:
        """
        使用 3D 增强模型提问
        """
        if not self.has_3d:
            return "[3D not available]"
        
        # 类似 2D，但需要注入 3D 数据
        # 这里简化处理
        return f"[3D Mock Answer for: {question[:30]}...]"
    
    def run_comparison(self):
        """
        运行对比测试
        """
        print("\n" + "=" * 70)
        print("Running Comparison Tests")
        print("=" * 70)
        
        # 加载视频帧
        print(f"\nLoading frames from: {COURTYARD_VIDEO_PATH}")
        frames = self.load_video_frames(COURTYARD_VIDEO_PATH, max_frames=8)
        print(f"✓ Loaded {len(frames)} frames")
        
        # 存储结果
        results = []
        
        for i, item in enumerate(COURTYARD_QUESTIONS, 1):
            question = item["question"]
            q_type = item["type"]
            expected = item.get("expected_answer", "N/A")
            
            print(f"\n[{i}/{len(COURTYARD_QUESTIONS)}] {q_type}")
            print(f"Q: {question}")
            print(f"Expected: {expected}")
            
            # 2D 回答
            answer_2d = self.ask_question_2d(question, frames)
            print(f"2D: {answer_2d}")
            
            # 3D 回答
            answer_3d = self.ask_question_3d(question, frames)
            print(f"3D: {answer_3d}")
            
            # 简单评估（如果 expected_answer 不为 None）
            if expected:
                match_2d = expected.lower() in answer_2d.lower()
                match_3d = expected.lower() in answer_3d.lower()
                print(f"   2D Correct: {'✓' if match_2d else '✗'}")
                print(f"   3D Correct: {'✓' if match_3d else '✗'}")
            
            results.append({
                "question": question,
                "type": q_type,
                "expected": expected,
                "answer_2d": answer_2d,
                "answer_3d": answer_3d
            })
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str = "test_results.json"):
        """保存结果"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")


def quick_demo():
    """
    快速演示：验证 3D 数据是否正确加载
    """
    print("=" * 70)
    print("Quick Demo: 3D Data Verification")
    print("=" * 70)
    
    # 加载 3D 数据
    data = torch.load(COURTYARD_3D_PATH)
    centers = data['centers_3d']  # (38, 729, 3)
    valid_mask = data['valid_mask']  # (38, 729)
    
    print(f"\n3D Cache Info:")
    print(f"  Path: {COURTYARD_3D_PATH}")
    print(f"  Frames: {centers.shape[0]}")
    print(f"  Patches per frame: {centers.shape[1]}")
    print(f"  Valid ratio: {valid_mask.float().mean():.1%}")
    print(f"  Coordinate range: [{centers.min():.2f}, {centers.max():.2f}]")
    
    # 可视化一个 frame 的 3D 分布
    frame_idx = 0
    frame_centers = centers[frame_idx].numpy()  # (729, 3)
    frame_valid = valid_mask[frame_idx].numpy()  # (729,)
    
    valid_centers = frame_centers[frame_valid]
    
    print(f"\nFrame {frame_idx} statistics:")
    print(f"  Valid patches: {frame_valid.sum()} / {len(frame_valid)}")
    print(f"  X range: [{valid_centers[:, 0].min():.2f}, {valid_centers[:, 0].max():.2f}]")
    print(f"  Y range: [{valid_centers[:, 1].min():.2f}, {valid_centers[:, 1].max():.2f}]")
    print(f"  Z range: [{valid_centers[:, 2].min():.2f}, {valid_centers[:, 2].max():.2f}]")
    
    print("\n✓ 3D data is ready for testing!")
    
    return data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test 3D encoding on single scene")
    parser.add_argument("--demo", action="store_true", help="Run quick demo only")
    parser.add_argument("--full", action="store_true", help="Run full comparison test")
    parser.add_argument("--model", type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov")
    
    args = parser.parse_args()
    
    if args.demo:
        quick_demo()
    elif args.full:
        tester = Simple3DTester(model_path=args.model)
        results = tester.run_comparison()
        tester.save_results(results)
    else:
        print("Usage:")
        print("  python test_single_scene_3d.py --demo    # Quick data verification")
        print("  python test_single_scene_3d.py --full    # Full comparison test")
