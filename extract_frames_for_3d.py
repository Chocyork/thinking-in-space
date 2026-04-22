#!/usr/bin/env python
"""
为 MapAnything 3D 重建切帧
============================
按照 VLM 同样的均匀采样逻辑提取帧，确保和推理时完全一致

使用方法:
    python extract_frames_for_3d.py \
        --video_dir ~/.cache/huggingface/vsibench/scannet \
        --output_dir /mnt/data/qyk/vsibench_frames/scannet \
        --num_frames 32
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from decord import VideoReader, cpu
from PIL import Image

VIDEO_DIRS = [
    ("arkitscenes", os.path.expanduser("~/.cache/huggingface/vsibench/arkitscenes")),
    ("scannet", os.path.expanduser("~/.cache/huggingface/vsibench/scannet")),
    ("scannetpp", os.path.expanduser("~/.cache/huggingface/vsibench/scannetpp")),
]


def extract_frames(video_path, output_dir, num_frames=32):
    """
    按照 VLM 同样的逻辑提取帧
    
    Returns:
        list: 提取的帧文件路径列表
    """
    video_name = Path(video_path).stem
    
    # 创建输出子目录
    frame_dir = os.path.join(output_dir, video_name)
    os.makedirs(frame_dir, exist_ok=True)
    
    # 检查是否已提取
    existing = sorted(Path(frame_dir).glob("frame_*.jpg"))
    if len(existing) == num_frames:
        return [str(f) for f in existing]
    
    try:
        # 打开视频
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frame_num = len(vr)
        
        # VLM 同样的采样逻辑: np.linspace(0, total_frame_num-1, num_frames, dtype=int)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, num_frames, dtype=int)
        frame_indices = uniform_sampled_frames.tolist()
        
        # 提取帧
        frames = vr.get_batch(frame_indices).asnumpy()  # (num_frames, H, W, C)
        
        # 保存为图片
        saved_paths = []
        for i, (idx, frame) in enumerate(zip(frame_indices, frames)):
            # 命名格式: frame_{帧索引:06d}.jpg
            frame_filename = f"frame_{idx:06d}.jpg"
            frame_path = os.path.join(frame_dir, frame_filename)
            
            # 转换为 PIL 并保存
            img = Image.fromarray(frame)
            img.save(frame_path, quality=95)
            saved_paths.append(frame_path)
        
        return saved_paths
        
    except Exception as e:
        print(f"✗ Error extracting {video_name}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["arkitscenes", "scannet", "scannetpp", "all"], 
                        default="all", help="要处理的数据集")
    parser.add_argument("--output_base", default="/mnt/data/qyk/vsibench_frames",
                        help="帧输出根目录")
    parser.add_argument("--num_frames", type=int, default=32,
                        help="每个视频提取的帧数（要和 VLM 的 max_frames_num 一致）")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理视频数量（测试用）")
    args = parser.parse_args()
    
    # 选择要处理的数据集
    if args.dataset == "all":
        datasets_to_process = VIDEO_DIRS
    else:
        datasets_to_process = [(args.dataset, dict(VIDEO_DIRS)[args.dataset])
                               if args.dataset in dict(VIDEO_DIRS) else []]
    
    print(f"{'='*70}")
    print("切帧用于 MapAnything 3D 重建")
    print(f"{'='*70}")
    print(f"采样帧数: {args.num_frames} (和 VLM max_frames_num 一致)")
    print(f"采样逻辑: np.linspace(0, total_frames-1, {args.num_frames}, dtype=int)")
    print(f"{'='*70}\n")
    
    total_videos = 0
    total_extracted = 0
    
    for dataset_name, video_dir in datasets_to_process:
        if not os.path.exists(video_dir):
            print(f"⚠ 目录不存在，跳过: {video_dir}")
            continue
        
        # 获取所有视频
        video_files = sorted(Path(video_dir).glob("*.mp4"))
        if args.limit:
            video_files = video_files[:args.limit]
        
        output_dir = os.path.join(args.output_base, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n[{dataset_name}] 找到 {len(video_files)} 个视频")
        print(f"输出目录: {output_dir}")
        
        # 处理每个视频
        for video_path in tqdm(video_files, desc=f"Extracting {dataset_name}"):
            saved = extract_frames(str(video_path), output_dir, args.num_frames)
            if len(saved) == args.num_frames:
                total_extracted += 1
            total_videos += 1
    
    print(f"\n{'='*70}")
    print("切帧完成！")
    print(f"{'='*70}")
    print(f"总视频数: {total_videos}")
    print(f"成功提取: {total_extracted}")
    print(f"预期总帧数: {total_extracted * args.num_frames}")
    print(f"\n下一步: 运行 MapAnything 处理这些帧生成 3D 缓存")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
