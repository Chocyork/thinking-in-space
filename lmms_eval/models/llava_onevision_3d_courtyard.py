"""
LLaVA-OneVision-3D (Courtyard Only)
====================================
仅用于测试单个 courtyard 视频的 3D 效果，不污染主代码
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from decord import VideoReader, cpu

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "map-anything"))
sys.path.insert(0, str(_PROJECT_ROOT / "thinking-in-space"))

# 继承自已有的 3D 类
from lmms_eval.models.llava_onevision_3d import Llava_OneVision_3D
from lmms_eval.api.registry import register_model


@register_model("llava_onevision_3d_courtyard")
class Llava_OneVision_3D_Courtyard(Llava_OneVision_3D):
    """
    LLaVA-OneVision-3D Courtyard 专用版本
    只处理 courtyard.mp4 视频，用于快速验证 3D 编码效果
    """
    
    def __init__(
        self,
        pretrained: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        use_3d: bool = False,
        point_cloud_path: str = None,
        **kwargs
    ):
        # 保存 courtyard 专用缓存路径
        self.courtyard_cache_file = None
        if use_3d and point_cloud_path:
            if os.path.isdir(point_cloud_path):
                courtyard_path = os.path.join(point_cloud_path, "courtyard_3d.pt")
                if os.path.exists(courtyard_path):
                    self.courtyard_cache_file = courtyard_path
                    print(f"[3D-Courtyard] Will use cache: {courtyard_path}")
        
        # 调用父类，保持 point_cloud_path 为目录
        super().__init__(
            pretrained=pretrained,
            use_3d=use_3d,
            point_cloud_path=point_cloud_path,
            **kwargs
        )
    
    def generate_until(self, requests):
        """
        重写生成方法，跳过非 courtyard 视频
        """
        if not self.use_3d:
            return super().generate_until(requests)
        
        # 过滤只保留 courtyard 相关的请求
        courtyard_requests = []
        for req in requests:
            contexts, doc_to_target, doc_to_visual, doc_id, task, split = req.args
            visual = doc_to_visual(self.task_dict[task][split][doc_id])
            
            if visual and isinstance(visual[0], str):
                video_path = visual[0]
                video_name = os.path.splitext(os.path.basename(video_path))[0].lower()
                
                # 检查是否是 courtyard 视频
                if "courtyard" in video_name:
                    courtyard_requests.append(req)
                    print(f"[3D-Courtyard] Processing: {video_name}")
                else:
                    print(f"[3D-Courtyard] Skipping: {video_name}")
        
        if not courtyard_requests:
            print("[3D-Courtyard] Warning: No courtyard videos found in requests!")
            # 返回空结果
            return [""] * len(requests)
        
        # 处理过滤后的请求
        if courtyard_requests:
            first_req = courtyard_requests[0]
            contexts, doc_to_target, doc_to_visual, doc_id, task, split = first_req.args
            visual = doc_to_visual(self.task_dict[task][split][doc_id])
            
            if visual and isinstance(visual[0], str):
                video_path = visual[0]
                # 使用 8 帧进行快速测试
                coords, mask = self._load_3d_for_video(video_path, 8)
                
                if coords is not None:
                    self._current_3d_coords = coords.view(-1, coords.shape[-2], coords.shape[-1])
                    self._current_3d_mask = mask.view(-1, mask.shape[-2], mask.shape[-1])
                    print(f"[3D-Courtyard] 3D data loaded: {coords.shape}")
                else:
                    print("[3D-Courtyard] Warning: Failed to load 3D data!")
        
        try:
            # 调用父类的 generate_until，但传入过滤后的请求
            results = super(Llava_OneVision_3D, self).generate_until(courtyard_requests)
        finally:
            if hasattr(self, '_current_3d_coords'):
                delattr(self, '_current_3d_coords')
            if hasattr(self, '_current_3d_mask'):
                delattr(self, '_current_3d_mask')
        
        # 将结果映射回原始请求（非 courtyard 的返回空字符串）
        full_results = []
        result_idx = 0
        for req in requests:
            contexts, doc_to_target, doc_to_visual, doc_id, task, split = req.args
            visual = doc_to_visual(self.task_dict[task][split][doc_id])
            
            if visual and isinstance(visual[0], str):
                video_path = visual[0]
                video_name = os.path.splitext(os.path.basename(video_path))[0].lower()
                
                if "courtyard" in video_name and result_idx < len(results):
                    full_results.append(results[result_idx])
                    result_idx += 1
                else:
                    full_results.append("")  # 非 courtyard 返回空
        
        return full_results
    
    def _load_3d_cache(self, cache_name: str):
        """
        覆盖父类方法，强制使用 courtyard 缓存
        """
        import os
        
        if self.courtyard_cache_file and os.path.exists(self.courtyard_cache_file):
            data_3d = torch.load(self.courtyard_cache_file, map_location='cpu')
            print(f"[3D-Courtyard] Loaded cache: {self.courtyard_cache_file} "
                  f"({data_3d['centers_3d'].shape[0]} frames)")
            return data_3d
        else:
            print(f"[3D-Courtyard] Warning: Cache file not found: {self.courtyard_cache_file}")
            return None
