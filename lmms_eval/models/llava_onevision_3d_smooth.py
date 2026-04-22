"""
LLaVA-OneVision with 3D Soft-Smoothing
======================================
在 projector 输出后、与文本合并前，对视觉特征做 soft-smoothing。
"""

import sys
import torch
import json
from pathlib import Path
from typing import List

sys.path.insert(0, "/home/qyk/map-anything")
sys.path.insert(0, "/home/qyk/thinking-in-space")
sys.path.insert(0, "LLaVA-NeXT/")

from lmms_eval.models.llava_onevision import Llava_OneVision
from lmms_eval.api.registry import register_model


# 全局配置
_matching_groups = {}
_alpha = 0.4


def _smooth_features(features, video_id):
    """核心 smoothing 函数 - 带2D空间感知的索引映射"""
    import math
    global _matching_groups, _alpha
    
    # 处理 List 类型输入（LLaVA 视频处理返回的是 List[Tensor]）
    is_list_input = isinstance(features, list)
    if is_list_input:
        if len(features) == 0:
            return features
        feat_to_process = features[0]
    else:
        feat_to_process = features
    
    if video_id not in _matching_groups:
        return features
    
    data = _matching_groups[video_id]
    groups = data.get("groups", [])
    num_frames = int(data.get("num_frames", 16))
    
    if not groups:
        return features
    
    # 处理形状 [B, N, D] 或 [num_frames, tokens_per_frame, hidden_dim] 或 [N, D]
    orig_shape = feat_to_process.shape
    needs_reshape_back = False
    
    if len(orig_shape) == 3:
        if orig_shape[0] == 1:
            # [1, N, D] -> [N, D]
            feat_to_process = feat_to_process.squeeze(0)
            total_tokens = feat_to_process.shape[0]
            tokens_per_frame = total_tokens // num_frames
        else:
            # [num_frames, tokens_per_frame, hidden_dim] -> [num_frames * tokens_per_frame, hidden_dim]
            num_frames_actual = orig_shape[0]
            tokens_per_frame = orig_shape[1]
            hidden_dim = orig_shape[2]
            total_tokens = num_frames_actual * tokens_per_frame
            feat_to_process = feat_to_process.view(-1, hidden_dim)
            needs_reshape_back = True
    elif len(orig_shape) == 2:
        # [total_tokens, hidden_dim]
        total_tokens = feat_to_process.shape[0]
        tokens_per_frame = total_tokens // num_frames
    else:
        return features
    
    # 2D 网格参数
    orig_grid_size = 27  # 原始 27x27 = 729 patches
    new_grid_size = int(math.sqrt(tokens_per_frame))  # 池化后的网格大小 (如 14x14)
    
    # 克隆避免修改原始张量
    smoothed = feat_to_process.clone()
    num_smoothed = 0
    total_members = 0
    
    # 遍历每个 group 做 soft-smoothing
    for group in groups:
        members = group.get("members", [])
        if len(members) < 2:
            continue
        
        # 收集 token 索引（使用2D空间感知映射）
        indices = []
        for m in members:
            frame = int(m["frame"])
            patch = int(m["patch"])  # 0-728
            
            # 1. 原 patch 1D -> 2D 坐标 (row, col)
            orig_row = patch // orig_grid_size  # 0-26
            orig_col = patch % orig_grid_size   # 0-26
            
            # 2. 按比例映射到新的 2D 网格
            new_row = int(orig_row * new_grid_size / orig_grid_size)
            new_col = int(orig_col * new_grid_size / orig_grid_size)
            
            # 边界检查
            new_row = min(new_row, new_grid_size - 1)
            new_col = min(new_col, new_grid_size - 1)
            
            # 3. 新 2D 坐标 -> 1D token 索引
            token_in_frame = new_row * new_grid_size + new_col
            
            # 4. 全局 token 索引
            idx = frame * tokens_per_frame + token_in_frame
            idx = min(idx, total_tokens - 1)
            indices.append(idx)
        
        indices = list(set(indices))
        if len(indices) < 2:
            continue
        
        # Soft-smoothing: new = (1-α)*original + α*mean
        with torch.no_grad():
            group_feats = smoothed[indices]
            mean_feat = group_feats.mean(dim=0, keepdim=True)
            for idx in indices:
                smoothed[idx] = (1 - _alpha) * smoothed[idx] + _alpha * mean_feat
        num_smoothed += 1
        total_members += len(indices)
    
    # 恢复原始形状
    if needs_reshape_back:
        # 从 [total_tokens, hidden_dim] 恢复为 [num_frames, tokens_per_frame, hidden_dim]
        smoothed = smoothed.view(orig_shape)
    
    # 如果是 list 输入，打包回 list 格式
    if is_list_input:
        return [smoothed]
    else:
        return smoothed


@register_model("llava_onevision_3d_smooth")
class Llava_OneVision_3D_Smooth(Llava_OneVision):
    """LLaVA-OneVision with 3D Soft-Smoothing"""
    
    def __init__(self, pretrained="lmms-lab/llava-onevision-qwen2-0.5b-ov",
                 matching_groups_path=None, alpha=0.4, **kwargs):
        
        # 先加载全局数据（每个进程都要加载）
        global _matching_groups, _alpha
        _alpha = alpha
        if matching_groups_path:
            mg_dir = Path(matching_groups_path)
            for json_file in mg_dir.glob("*/matching_groups.json"):
                vid = json_file.parent.name
                try:
                    with open(json_file) as f:
                        _matching_groups[vid] = json.load(f)
                except Exception as e:
                    pass
        
        # 父类初始化
        super().__init__(pretrained=pretrained, **kwargs)
        
        # 关键：替换 encode_images，在 projector 后做 smoothing
        self._original_encode = self.model.encode_images
        self._matching_groups_path = matching_groups_path
        self._setup_encode_wrapper()
        
        print(f"[3D-Smooth] 初始化完成: 加载了 {len(_matching_groups)} 个视频的 matching groups, alpha={alpha}")
    
    def _setup_encode_wrapper(self):
        """包装 encode_images，在 projector 后插入 smoothing"""
        import types
        orig = self._original_encode
        mg_path = self._matching_groups_path
        
        def wrapped_encode(self_model, images):
            global _matching_groups, _alpha
            
            # 1. 原始编码 (Vision Tower + Projector)
            feats = orig(images)
            
            # 2. 获取当前视频 ID
            import os
            vid = os.environ.get('SMOOTH_CURR_VIDEO', '')
            
            # 3. 子进程中如果没有数据，重新加载
            if not _matching_groups and mg_path:
                mg_dir = Path(mg_path)
                for json_file in mg_dir.glob("*/matching_groups.json"):
                    v = json_file.parent.name
                    try:
                        with open(json_file) as f:
                            _matching_groups[v] = json.load(f)
                    except:
                        pass
            
            # 4. 应用 smoothing
            if vid and vid in _matching_groups:
                feats = _smooth_features(feats, vid)
            
            return feats
        
        self.model.encode_images = types.MethodType(wrapped_encode, self.model)
    
    def generate_until(self, requests: List[str]) -> List[str]:
        """处理请求前设置 video_id"""
        import os
        
        # 提取 video_id
        vid = None
        if requests:
            try:
                args = requests[0].args
                if len(args) >= 6:
                    _, _, doc_to_visual, doc_id, task, split = args
                    
                    if hasattr(self, 'task_dict') and task in self.task_dict and split in self.task_dict[task]:
                        doc = self.task_dict[task][split][doc_id]
                        paths = doc_to_visual(doc)
                        
                        if paths:
                            import os as os2
                            # 提取完整 video_id: 文件夹名_视频名 (如 arkitscenes_42445981)
                            parent_dir = os2.path.basename(os2.path.dirname(paths[0]))
                            filename = os2.path.basename(paths[0]).replace('.mp4', '')
                            vid = f"{parent_dir}_{filename}"
            except Exception as e:
                pass
        
        # 设置环境变量（所有子进程可见）
        if vid:
            os.environ['SMOOTH_CURR_VIDEO'] = vid
        
        # 调用父类生成
        try:
            results = super().generate_until(requests)
        finally:
            os.environ.pop('SMOOTH_CURR_VIDEO', None)
        
        return results
