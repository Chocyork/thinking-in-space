"""
LLaVA-OneVision with 3D Salience-Based Pruning (Max Norm)
"""

import sys
import torch
import json
import math
from pathlib import Path
from typing import List

sys.path.insert(0, "/home/qyk/map-anything")
sys.path.insert(0, "/home/qyk/thinking-in-space")
sys.path.insert(0, "LLaVA-NeXT/")

from lmms_eval.models.llava_onevision import Llava_OneVision
from lmms_eval.api.registry import register_model

_matching_groups = {}
# 保留alpha参数但不再使用（为了兼容性）
_alpha = 0.4

@register_model("llava_onevision_3d_prune_max")
class Llava_OneVision_3D_Prune_Max(Llava_OneVision):
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
        
        self._setup_prepare_wrapper()
        import sys
        print(f"\n🚀 [3D剪枝系统启动] 成功加载了 {len(_matching_groups)} 个视频的缓存！Alpha={_alpha}\n", file=sys.stderr, flush=True)

    def _setup_prepare_wrapper(self):
        import types
        import os
        import sys
        orig_prepare = self._original_prepare
        
        def wrapped_prepare(self_model, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes=None, **kwargs):
            res = orig_prepare(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, **kwargs)
            ret_inputs, ret_pos, ret_attn, ret_past, ret_embeds, ret_labels = res
            
            vid = os.environ.get('SMOOTH_CURR_VIDEO', '')
            if not vid or vid not in _matching_groups or ret_embeds is None or input_ids is None:
                return res
                
            pos = (input_ids[0] == -200).nonzero(as_tuple=True)[0]
            if len(pos) == 0: 
                pos = (input_ids[0] == getattr(self_model.config, 'image_token_index', 151646)).nonzero(as_tuple=True)[0]
            
            if len(pos) > 0:
                # [关键改动1] 放弃错误的相减，改用夹逼法提取纯视觉长度
                prefix_len = pos[0].item()
                suffix_len = input_ids.shape[1] - pos[-1].item() - 1
                vis_len = ret_embeds.shape[1] - prefix_len - suffix_len
                
                num_frames = int(_matching_groups[vid].get("num_frames", 16))
                # if vis_len % num_frames != 0: 
                #     print(f"\n⚠️ [放弃剪枝-帧数除不尽] 视频 {vid} | 纯视觉Token总数 {vis_len} 无法被 {num_frames} 帧平分！", file=sys.stderr, flush=True)
                #     return res # 安全跳过，防止崩溃
                    
                tokens_per_frame = vis_len // num_frames
                
                # [关键改动2] 完美网格推导，避开 <newline> 误杀
                H = int((-1 + math.sqrt(1 + 4 * tokens_per_frame)) / 2)
                has_newlines = (H * (H + 1) == tokens_per_frame)
                if not has_newlines:
                    H = int(math.sqrt(tokens_per_frame))
                    if H * H != tokens_per_frame:
                        print(f"\n⚠️ [放弃剪枝-网格无法开方] 视频 {vid} | 单帧Token数 {tokens_per_frame} 不是标准网格！", file=sys.stderr, flush=True)
                        return res
                
                keep_mask = torch.ones(vis_len, dtype=torch.bool, device=ret_embeds.device)
                orig_grid = 27
                
                for group in _matching_groups[vid].get("groups", []):
                    members = group.get("members", [])
                    if len(members) < 2: continue
                    
                    group_idx = []
                    for m in members:
                        f, p = int(m["frame"]), int(m["patch"])
                        # 精准 2D 坐标映射，避开行末换行符
                        new_r = min(int((p // orig_grid) * H / orig_grid), H - 1)
                        new_c = min(int((p % orig_grid) * H / orig_grid), H - 1)
                        idx = f * tokens_per_frame + new_r * (H + 1 if has_newlines else H) + new_c
                        if idx not in group_idx: group_idx.append(idx)
                        
                    if len(group_idx) >= 2:
                        # 显著性保留：计算L2范数，保留最大的那个
                        start_idx = prefix_len
                        vis_embeds = ret_embeds[0, start_idx : start_idx + vis_len]  # [vis_len, hidden_dim]
                        group_feats = vis_embeds[group_idx]  # [num_members, hidden_dim]
                        norms = torch.norm(group_feats, dim=-1)  # L2范数
                        best_local_idx = torch.argmax(norms).item()  # 局部索引
                        best_idx = group_idx[best_local_idx]  # 全局索引
                        # 只保留best_idx，其他全部丢弃
                        for idx in group_idx:
                            if idx != best_idx:
                                keep_mask[idx] = False
                            
                B = ret_embeds.shape[0]
                start_idx = prefix_len
                expected_full_len = prefix_len + vis_len + suffix_len
                
                # 1. 切特征 (ret_embeds)
                vis_emb_pruned = ret_embeds[:, start_idx : start_idx + vis_len][:, keep_mask]
                ret_embeds = torch.cat([ret_embeds[:, :start_idx], vis_emb_pruned, ret_embeds[:, start_idx + vis_len :]], dim=1)
                
                # 2. 切文本 IDs (ret_inputs)
                if ret_inputs is not None and ret_inputs.shape[1] == expected_full_len:
                    vis_in_pruned = ret_inputs[:, start_idx : start_idx + vis_len][:, keep_mask]
                    ret_inputs = torch.cat([ret_inputs[:, :start_idx], vis_in_pruned, ret_inputs[:, start_idx + vis_len :]], dim=1)
                    
                # 3. 切标签 (ret_labels)
                if ret_labels is not None and ret_labels.shape[1] == expected_full_len:
                    vis_lab_pruned = ret_labels[:, start_idx : start_idx + vis_len][:, keep_mask]
                    ret_labels = torch.cat([ret_labels[:, :start_idx], vis_lab_pruned, ret_labels[:, start_idx + vis_len :]], dim=1)
                
                # [关键改动3] 原位切片 Position IDs！不再使用 torch.arange 连续重排
                if ret_pos is not None and ret_pos.shape[1] == expected_full_len:
                    vis_pos_pruned = ret_pos[:, start_idx : start_idx + vis_len][:, keep_mask]
                    ret_pos = torch.cat([ret_pos[:, :start_idx], vis_pos_pruned, ret_pos[:, start_idx + vis_len :]], dim=1)
                
                # Attention Mask 保持全 1 即可
                if ret_attn is not None:
                    new_len = ret_embeds.shape[1]
                    ret_attn = torch.ones((B, new_len), dtype=torch.bool, device=ret_embeds.device)
                
                # 强制调试打印 (确认剪枝生效，后续嫌烦可注释)
                #print(f"\n🔥 [原位保留剪枝成功] 视频 {vid} | 同步切片: {vis_len} -> {vis_emb_pruned.shape[1]}\n", file=sys.stderr, flush=True)
                
            return ret_inputs, ret_pos, ret_attn, ret_past, ret_embeds, ret_labels

        self.model.prepare_inputs_labels_for_multimodal = types.MethodType(wrapped_prepare, self.model)


    def generate_until(self, requests: List[str]) -> List[str]:
        import os
        import sys
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
                            vid = f"{os2.path.basename(os2.path.dirname(paths[0]))}_{os2.path.basename(paths[0]).replace('.mp4', '')}"
            except Exception as e: 
                print(f"❌ [VID提取崩溃]: {e}", file=sys.stderr, flush=True)
                
        # === 核心喇叭：无论成功还是失败，都必须吼一声！ ===
        if vid: 
            print(f"✅ [准备推理] 成功锁定当前视频: {vid}", file=sys.stderr, flush=True)
            os.environ['SMOOTH_CURR_VIDEO'] = vid
        else:
            print(f"⚠️ [警告] 视频ID提取失败！本题将完全不触发剪枝", file=sys.stderr, flush=True)
        # ===================================================
            
        try:
            return super().generate_until(requests)
        finally:
            os.environ.pop('SMOOTH_CURR_VIDEO', None)