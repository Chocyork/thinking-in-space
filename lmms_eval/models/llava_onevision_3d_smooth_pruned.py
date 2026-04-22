"""
LLaVA-OneVision with 3D Soft-Smoothing + Delayed Smart Pruning
"""

import sys
import torch
import json
import math
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "map-anything"))
sys.path.insert(0, str(_PROJECT_ROOT / "thinking-in-space"))
sys.path.insert(0, str(_PROJECT_ROOT / "thinking-in-space" / "LLaVA-NeXT"))

from lmms_eval.models.llava_onevision import Llava_OneVision
from lmms_eval.api.registry import register_model

_matching_groups = {}
_alpha = 0.4

def _smooth_features(features, video_id):
    """阶段1: 纯特征平滑 (染色)，不物理删除"""
    global _matching_groups, _alpha
    
    is_list_input = isinstance(features, list)
    feat_to_process = features[0] if is_list_input and len(features)>0 else features
    if video_id not in _matching_groups or not _matching_groups[video_id].get("groups"):
        return features
        
    groups = _matching_groups[video_id]["groups"]
    num_frames = int(_matching_groups[video_id].get("num_frames", 16))
    
    orig_shape = feat_to_process.shape
    needs_reshape_back = False
    
    if len(orig_shape) == 3:
        if orig_shape[0] == 1:
            feat_to_process = feat_to_process.squeeze(0)
        else:
            feat_to_process = feat_to_process.view(-1, orig_shape[2])
            needs_reshape_back = True
            
    total_tokens = feat_to_process.shape[0]
    tokens_per_frame = total_tokens // num_frames
    
    orig_grid_size = 27
    new_grid_size = int(math.sqrt(tokens_per_frame))
    smoothed = feat_to_process.clone()
    
    for group in groups:
        members = group.get("members", [])
        if len(members) < 2: continue
        
        indices = []
        for m in members:
            frame = int(m["frame"])
            patch = int(m["patch"])
            orig_r, orig_c = patch // orig_grid_size, patch % orig_grid_size
            new_r = min(int(orig_r * new_grid_size / orig_grid_size), new_grid_size - 1)
            new_c = min(int(orig_c * new_grid_size / orig_grid_size), new_grid_size - 1)
            idx = frame * tokens_per_frame + new_r * new_grid_size + new_c
            indices.append(min(idx, total_tokens - 1))
            
        indices = list(set(indices))
        if len(indices) >= 2:
            with torch.no_grad():
                mean_feat = smoothed[indices].mean(dim=0)
                smoothed[indices[0]] = (1 - _alpha) * smoothed[indices[0]] + _alpha * mean_feat
                
    if needs_reshape_back:
        smoothed = smoothed.view(orig_shape)
    return [smoothed] if is_list_input else smoothed


@register_model("llava_onevision_3d_smooth_pruned")
class Llava_OneVision_3D_Smooth_Pruned(Llava_OneVision):
    def __init__(self, pretrained="lmms-lab/llava-onevision-qwen2-0.5b-ov", matching_groups_path=None, alpha=0.4, **kwargs):
        global _matching_groups, _alpha
        _alpha = alpha
        if matching_groups_path:
            for json_file in Path(matching_groups_path).glob("*/matching_groups.json"):
                try:
                    with open(json_file) as f:
                        _matching_groups[json_file.parent.name] = json.load(f)
                except: pass
                
        super().__init__(pretrained=pretrained, **kwargs)
        self._original_encode = self.model.encode_images
        self._original_prepare = self.model.prepare_inputs_labels_for_multimodal
        self._matching_groups_path = matching_groups_path
        
        self._setup_encode_wrapper()
        self._setup_prepare_wrapper()

    def _setup_encode_wrapper(self):
        import types
        import os
        orig = self._original_encode
        def wrapped_encode(self_model, images):
            feats = orig(images)
            vid = os.environ.get('SMOOTH_CURR_VIDEO', '')
            if vid and vid in _matching_groups:
                feats = _smooth_features(feats, vid)
            return feats
        self.model.encode_images = types.MethodType(wrapped_encode, self.model)

    def _setup_prepare_wrapper(self):
        import types
        import os
        import sys
        orig_prepare = self._original_prepare
        
        def wrapped_prepare(self_model, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes=None, **kwargs):
            # 1. 拿到底层拼接好的完整长张量
            res = orig_prepare(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, **kwargs)
            ret_inputs, ret_pos, ret_attn, ret_past, ret_embeds, ret_labels = res
            
            vid = os.environ.get('SMOOTH_CURR_VIDEO', '')
            if not vid or vid not in _matching_groups or ret_embeds is None or input_ids is None:
                return res
                
            # 2. 定位视觉特征在超长序列里的起点
            pos = (input_ids[0] == -200).nonzero(as_tuple=True)[0]
            if len(pos) == 0: 
                pos = (input_ids[0] == getattr(self_model.config, 'image_token_index', 151646)).nonzero(as_tuple=True)[0]
            
            if len(pos) > 0:
                start_idx = pos[0].item()
                vis_len = ret_embeds.shape[1] - input_ids.shape[1] + 1
                
                # 3. 极简 EVS 剪枝策略：线性映射 + 生成掩码
                num_frames = int(_matching_groups[vid].get("num_frames", 16))
                tokens_per_frame = vis_len // num_frames
                
                keep_mask = torch.ones(vis_len, dtype=torch.bool, device=ret_embeds.device)
                
                for group in _matching_groups[vid].get("groups", []):
                    members = group.get("members", [])
                    if len(members) < 2: continue
                    
                    group_idx = []
                    for m in members:
                        f = int(m["frame"])
                        p = int(m["patch"]) # 原 MapAnything 的 patch 索引 (0-728)
                        
                        mapped_p = min(int((p / 729) * tokens_per_frame), tokens_per_frame - 1)
                        idx = f * tokens_per_frame + mapped_p
                        if idx not in group_idx: group_idx.append(idx)
                        
                    if len(group_idx) >= 2:
                        for drop_idx in group_idx[1:]:
                            keep_mask[drop_idx] = False
                            
                # ==========================================================
                # 4. 终极修复：绝对同步切片！不仅切特征，连同 ID 和标签一起切！
                # ==========================================================
                B = ret_embeds.shape[0]
                
                # 切特征 (ret_embeds)
                vis_emb_pruned = ret_embeds[:, start_idx : start_idx + vis_len][:, keep_mask]
                ret_embeds = torch.cat([ret_embeds[:, :start_idx], vis_emb_pruned, ret_embeds[:, start_idx + vis_len :]], dim=1)
                
                # 切文本 IDs (ret_inputs) - 修复崩溃的元凶！
                if ret_inputs is not None:
                    vis_in_pruned = ret_inputs[:, start_idx : start_idx + vis_len][:, keep_mask]
                    ret_inputs = torch.cat([ret_inputs[:, :start_idx], vis_in_pruned, ret_inputs[:, start_idx + vis_len :]], dim=1)
                    
                # 切标签 (ret_labels)
                if ret_labels is not None:
                    vis_lab_pruned = ret_labels[:, start_idx : start_idx + vis_len][:, keep_mask]
                    ret_labels = torch.cat([ret_labels[:, :start_idx], vis_lab_pruned, ret_labels[:, start_idx + vis_len :]], dim=1)
                
                # 同步重塑 Position IDs 和 Attention Mask
                new_len = ret_embeds.shape[1]
                if ret_pos is not None:
                    ret_pos = torch.arange(new_len, dtype=torch.long, device=ret_embeds.device).unsqueeze(0).expand(B, -1)
                if ret_attn is not None:
                    ret_attn = torch.ones((B, new_len), dtype=torch.bool, device=ret_embeds.device)
                
                # print(f"\n🔥 [EVS剪枝成功] 视频 {vid} | 同步切片: {vis_len} -> {vis_emb_pruned.shape[1]}\n", file=sys.stderr, flush=True)
                
            return ret_inputs, ret_pos, ret_attn, ret_past, ret_embeds, ret_labels

        self.model.prepare_inputs_labels_for_multimodal = types.MethodType(wrapped_prepare, self.model)


    def generate_until(self, requests: List[str]) -> List[str]:
        import os
        vid = None
        if requests:
            try:
                args = requests[0].args
                if len(args) >= 6:
                    _, _, doc_to_visual, doc_id, task, split = args
                    doc = self.task_dict[task][split][doc_id]
                    paths = doc_to_visual(doc)
                    if paths:
                        import os as os2
                        vid = f"{os2.path.basename(os2.path.dirname(paths[0]))}_{os2.path.basename(paths[0]).replace('.mp4', '')}"
            except: pass
        if vid: os.environ['SMOOTH_CURR_VIDEO'] = vid
        try:
            return super().generate_until(requests)
        finally:
            os.environ.pop('SMOOTH_CURR_VIDEO', None)
