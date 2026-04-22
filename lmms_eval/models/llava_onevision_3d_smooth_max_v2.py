"""
LLaVA-OneVision with 3D Smoothing + Max Norm Pruning (Stage-2 Unified)
所有操作在 prepare_inputs_labels 中统一完成：
1. 计算 L2 找出每组 best token
2. Smoothing: 把组内均值融合到 best token
3. Pruning: 删除其他 token
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


@register_model("llava_onevision_3d_smooth_max_v2")
class Llava_OneVision_3D_Smooth_Max_v2(Llava_OneVision):
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
        self._original_prepare = self.model.prepare_inputs_labels_for_multimodal
        self._matching_groups_path = matching_groups_path
        
        self._setup_prepare_wrapper()
        import sys
        # print(f"\n🚀 [3D Smooth+MaxPrune v2 启动] 加载了 {len(_matching_groups)} 个视频！Alpha={_alpha}\n", file=sys.stderr, flush=True)

    def _setup_prepare_wrapper(self):
        import types
        import os
        import sys
        orig_prepare = self._original_prepare
        
        def wrapped_prepare(self_model, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes=None, **kwargs):
            # 1. 获取 embeddings
            res = orig_prepare(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, **kwargs)
            ret_inputs, ret_pos, ret_attn, ret_past, ret_embeds, ret_labels = res
            
            vid = os.environ.get('SMOOTH_CURR_VIDEO', '')
            if not vid or vid not in _matching_groups or ret_embeds is None or input_ids is None:
                return res
                
            pos = (input_ids[0] == -200).nonzero(as_tuple=True)[0]
            if len(pos) == 0: 
                pos = (input_ids[0] == getattr(self_model.config, 'image_token_index', 151646)).nonzero(as_tuple=True)[0]
            
            if len(pos) > 0:
                # print(f"\n📍 [视频 {vid}] 找到 {len(pos)} 个视觉token", file=sys.stderr, flush=True)
                prefix_len = pos[0].item()
                suffix_len = input_ids.shape[1] - pos[-1].item() - 1
                vis_len = ret_embeds.shape[1] - prefix_len - suffix_len
                # print(f"   视觉长度: {vis_len} (prefix={prefix_len}, suffix={suffix_len})", file=sys.stderr, flush=True)
                
                num_frames = int(_matching_groups[vid].get("num_frames", 16))
                # 允许不整除，使用近似值
                tokens_per_frame = vis_len // num_frames
                # if vis_len % num_frames != 0:
                #     print(f"   ⚠️ 视觉长度 {vis_len} 不能被 {num_frames} 整除，使用近似 tokens_per_frame={tokens_per_frame}", file=sys.stderr, flush=True)
                
                H = int((-1 + math.sqrt(1 + 4 * tokens_per_frame)) / 2)
                has_newlines = (H * (H + 1) == tokens_per_frame)
                if not has_newlines:
                    H = int(math.sqrt(tokens_per_frame))
                    if H * H != tokens_per_frame: return res
                
                keep_mask = torch.ones(vis_len, dtype=torch.bool, device=ret_embeds.device)
                orig_grid = 27
                
                # 提取视觉特征区域用于处理
                start_idx = prefix_len
                vis_embeds = ret_embeds[0, start_idx : start_idx + vis_len]  # [vis_len, hidden_dim]
                
                processed_groups = 0
                pruned_tokens = 0
                
                for group in _matching_groups[vid].get("groups", []):
                    members = group.get("members", [])
                    if len(members) < 2: continue
                    
                    group_idx = []
                    for m in members:
                        f, p = int(m["frame"]), int(m["patch"])
                        new_r = min(int((p // orig_grid) * H / orig_grid), H - 1)
                        new_c = min(int((p % orig_grid) * H / orig_grid), H - 1)
                        idx = f * tokens_per_frame + new_r * (H + 1 if has_newlines else H) + new_c
                        if idx not in group_idx: group_idx.append(idx)
                        
                    if len(group_idx) >= 2:
                        # Step 1: 计算 L2，找出 best_idx
                        group_feats = vis_embeds[group_idx]  # [num_members, hidden_dim]
                        norms = torch.norm(group_feats, dim=-1)
                        best_local_idx = torch.argmax(norms).item()
                        best_idx = group_idx[best_local_idx]
                        
                        # Step 2: Smoothing - 把组内均值融合到 best_idx
                        with torch.no_grad():
                            mean_feat = vis_embeds[group_idx].mean(dim=0)
                            # 直接在 vis_embeds 上修改
                            vis_embeds[best_idx] = (1 - _alpha) * vis_embeds[best_idx] + _alpha * mean_feat
                        
                        # Step 3: Pruning - 标记删除其他
                        drop_count = 0
                        for idx in group_idx:
                            if idx != best_idx:
                                keep_mask[idx] = False
                                drop_count += 1
                        processed_groups += 1
                        pruned_tokens += drop_count
                
                # print(f"   处理了 {processed_groups} 个组，剪枝 {pruned_tokens} 个tokens", file=sys.stderr, flush=True)
                
                # 把修改后的 vis_embeds 写回 ret_embeds
                ret_embeds[0, start_idx : start_idx + vis_len] = vis_embeds
                
                # 应用 keep_mask 切片
                B = ret_embeds.shape[0]
                expected_full_len = prefix_len + vis_len + suffix_len
                
                vis_emb_pruned = ret_embeds[:, start_idx : start_idx + vis_len][:, keep_mask]
                ret_embeds = torch.cat([ret_embeds[:, :start_idx], vis_emb_pruned, ret_embeds[:, start_idx + vis_len :]], dim=1)
                
                if ret_inputs is not None and ret_inputs.shape[1] == expected_full_len:
                    vis_in_pruned = ret_inputs[:, start_idx : start_idx + vis_len][:, keep_mask]
                    ret_inputs = torch.cat([ret_inputs[:, :start_idx], vis_in_pruned, ret_inputs[:, start_idx + vis_len :]], dim=1)
                    
                if ret_labels is not None and ret_labels.shape[1] == expected_full_len:
                    vis_lab_pruned = ret_labels[:, start_idx : start_idx + vis_len][:, keep_mask]
                    ret_labels = torch.cat([ret_labels[:, :start_idx], vis_lab_pruned, ret_labels[:, start_idx + vis_len :]], dim=1)
                
                if ret_pos is not None and ret_pos.shape[1] == expected_full_len:
                    vis_pos_pruned = ret_pos[:, start_idx : start_idx + vis_len][:, keep_mask]
                    ret_pos = torch.cat([ret_pos[:, :start_idx], vis_pos_pruned, ret_pos[:, start_idx + vis_len :]], dim=1)
                
                if ret_attn is not None:
                    new_len = ret_embeds.shape[1]
                    ret_attn = torch.ones((B, new_len), dtype=torch.bool, device=ret_embeds.device)
                
                # print(f"\n🔥 [Smooth+MaxPrune v2] 视频 {vid} | {vis_len} -> {vis_emb_pruned.shape[1]}\n", file=sys.stderr, flush=True)
                
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
                    doc = self.task_dict[task][split][doc_id]
                    paths = doc_to_visual(doc)
                    if paths:
                        import os as os2
                        vid = f"{os2.path.basename(os2.path.dirname(paths[0]))}_{os2.path.basename(paths[0]).replace('.mp4', '')}"
            except Exception as e:
                print(f"   ⚠️ 视频ID提取失败: {e}", file=sys.stderr, flush=True)
                pass
        if vid:
            print(f"\n🎬 [准备推理] 视频ID: {vid}", file=sys.stderr, flush=True)
            # 检查是否在 matching_groups 中
            if vid in _matching_groups:
                groups_count = len(_matching_groups[vid].get("groups", []))
                print(f"   ✅ 找到 {groups_count} 个匹配组", file=sys.stderr, flush=True)
            else:
                print(f"   ❌ 该视频不在 matching_groups 中！可用视频数: {len(_matching_groups)}", file=sys.stderr, flush=True)
            os.environ['SMOOTH_CURR_VIDEO'] = vid
        else:
            print(f"\n⚠️ [警告] 无法提取视频ID，剪枝将不生效", file=sys.stderr, flush=True)
        try:
            return super().generate_until(requests)
        finally:
            # if vid:
            #     print(f"\n✅ [完成] 视频 {vid} 推理结束\n", file=sys.stderr, flush=True)
            os.environ.pop('SMOOTH_CURR_VIDEO', None)
