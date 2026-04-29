#!/usr/bin/env python3
"""
Profile per-video inference latency for VSI-Bench.

This script measures the current code path:

Baseline:
    T_baseline_video = N * T_baseline_inference_per_question

Prune:
    T_prune_video = T_reconstruction_once
                  + T_matching_once
                  + N * T_pruned_inference_per_question

where N is the number of questions for the selected video.

Run from the thinking-in-space repository root.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


DEFAULT_BASELINE_MODEL = "llava_onevision"
DEFAULT_PRUNE_MODEL = "llava_onevision_3d_prune_max"
DEFAULT_MATCHING_GROUPS = "/home/qyk/thinking-in-space/validation/43d3dpt_16f_match_voxel10cm_iou45"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="vsibench", help="lmms_eval task name.")
    parser.add_argument(
        "--variant",
        choices=["baseline", "prune", "both"],
        default="both",
        help="Which model path to profile. Use one variant per process for the cleanest GPU memory behavior.",
    )
    parser.add_argument("--pretrained", default="", help="Local model path or HF model id. Required unless --list-videos is used.")
    parser.add_argument("--video-key", default=None, help="Video key as 'dataset/scene_name' or 'dataset_scene_name'.")
    parser.add_argument("--video-index", type=int, default=0, help="Video index after sorting keys, used if --video-key is omitted.")
    parser.add_argument("--num-videos", type=int, default=1, help="Profile this many videos starting at --video-index.")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit questions per selected video.")
    parser.add_argument("--list-videos", action="store_true", help="List videos and question counts, then exit.")
    parser.add_argument("--list-limit", type=int, default=20, help="How many videos to show with --list-videos.")
    parser.add_argument("--device", default="cuda:0", help="Device string passed to lmms_eval model creation.")
    parser.add_argument("--batch-size", default="1", help="Batch size passed to lmms_eval model creation. LLaVA expects 1.")
    parser.add_argument("--max-frames-num", type=int, default=16)
    parser.add_argument("--mm-spatial-pool-stride", type=int, default=2)
    parser.add_argument("--conv-template", default="qwen_1_5")
    parser.add_argument("--model-name", default="llava_qwen")
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--prune-model", default=DEFAULT_PRUNE_MODEL)
    parser.add_argument("--matching-groups-path", default=DEFAULT_MATCHING_GROUPS)
    parser.add_argument("--alpha", default="0.4", help="Compatibility arg for prune models.")
    parser.add_argument("--extra-model-args", default="", help="Extra comma-separated model args appended to both variants.")
    parser.add_argument("--baseline-extra-model-args", default="", help="Extra args appended only to baseline.")
    parser.add_argument("--prune-extra-model-args", default="", help="Extra args appended only to prune.")
    parser.add_argument("--reconstruction-sec", type=float, default=0.0, help="Measured one-time reconstruction cost for the video.")
    parser.add_argument("--matching-sec", type=float, default=0.0, help="Measured one-time matching cost for the video.")
    parser.add_argument("--warmup-questions", type=int, default=0, help="Run this many questions before timing each variant.")
    parser.add_argument(
        "--measure-ttft",
        action="store_true",
        help="Measure time to first generated token by temporarily appending a non-stopping generation criterion.",
    )
    parser.add_argument("--include-responses", action="store_true", help="Store model responses in the JSON output.")
    parser.add_argument("--output-json", default=None, help="Output JSON path. Defaults to stdout only.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def setup_import_path() -> None:
    root = repo_root()
    os.environ.setdefault("LMMS_EVAL_LAUNCHER", "python")
    paths = [str(root), str(root / "LLaVA-NeXT")]
    old_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = ":".join(paths + ([old_pythonpath] if old_pythonpath else []))
    for path in reversed(paths):
        if path not in os.sys.path:
            os.sys.path.insert(0, path)


def load_task(task_name: str, model_name: str):
    from lmms_eval.tasks import TaskManager, get_task_dict

    manager = TaskManager("INFO", model_name=model_name)
    task_dict = get_task_dict([task_name], manager)
    task = task_dict[task_name]
    if isinstance(task, tuple):
        _, task = task
    return task


def get_split(task) -> str:
    if task.has_test_docs():
        return task.config.test_split
    if task.has_validation_docs():
        return task.config.validation_split
    raise RuntimeError("Task has neither test nor validation docs.")


def iter_docs(task, split: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    dataset = task.dataset[split]
    for doc_id in range(len(dataset)):
        yield doc_id, dataset[doc_id]


def doc_video_key(doc: Dict[str, Any]) -> str:
    dataset = str(doc.get("dataset", ""))
    scene_name = str(doc.get("scene_name", ""))
    return f"{dataset}/{scene_name}"


def matching_video_id(video_key: str) -> str:
    return video_key.replace("/", "_")


def group_docs_by_video(task, split: str) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for doc_id, doc in iter_docs(task, split):
        groups[doc_video_key(doc)].append(doc_id)
    return dict(groups)


def normalize_video_key(raw_key: str, available: Dict[str, List[int]]) -> str:
    if raw_key in available:
        return raw_key
    if "/" not in raw_key:
        matches = [key for key in available if matching_video_id(key) == raw_key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous video key {raw_key!r}; matches: {matches[:10]}")
    raise ValueError(f"Video key {raw_key!r} not found.")


def select_video_keys(args: argparse.Namespace, groups: Dict[str, List[int]]) -> List[str]:
    keys = sorted(groups)
    if args.video_key:
        start_key = normalize_video_key(args.video_key, groups)
        start_idx = keys.index(start_key)
    else:
        start_idx = args.video_index
    return keys[start_idx : start_idx + args.num_videos]


def construct_instance(task, doc_id: int, split: str):
    num_fewshot = 0 if task.config.num_fewshot is None else task.config.num_fewshot
    fewshot_split = task.config.training_split if task.has_training_docs() else split
    ctx = task.fewshot_context(doc_id, num_fewshot, fewshot_split)
    metadata = {"task": task.config["task"], "doc_id": doc_id, "repeats": task.config.repeats}
    if task.config.metadata and isinstance(task.config.metadata, dict):
        metadata.update(task.config.metadata)
    return task.construct_requests(doc_id=doc_id, ctx=ctx, metadata=metadata, split=split)


def create_model(model_key: str, model_args: str, batch_size: str, device: str):
    from lmms_eval.models import get_model

    model_cls = get_model(model_key)
    return model_cls.create_from_arg_string(
        model_args,
        {
            "batch_size": batch_size,
            "device": device,
        },
    )


def join_model_args(parts: List[str]) -> str:
    return ",".join([part for part in parts if part])


def common_model_args(args: argparse.Namespace) -> str:
    return join_model_args(
        [
            f"pretrained={args.pretrained}",
            f"conv_template={args.conv_template}",
            f"model_name={args.model_name}",
            f"max_frames_num={args.max_frames_num}",
            f"mm_spatial_pool_stride={args.mm_spatial_pool_stride}",
            args.extra_model_args,
        ]
    )


def baseline_model_args(args: argparse.Namespace) -> str:
    return join_model_args([common_model_args(args), args.baseline_extra_model_args])


def prune_model_args(args: argparse.Namespace) -> str:
    return join_model_args(
        [
            common_model_args(args),
            f"matching_groups_path={args.matching_groups_path}",
            f"alpha={args.alpha}",
            args.prune_extra_model_args,
        ]
    )


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TTFTStoppingCriteria:
    """Records the first generation step without changing model stopping behavior."""

    def __init__(self, start_sec: float):
        self.start_sec = start_sec
        self.first_token_sec: Optional[float] = None
        self.generated_steps = 0

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        self.generated_steps += 1
        if self.first_token_sec is None:
            cuda_sync()
            self.first_token_sec = time.perf_counter() - self.start_sec
        return False


def append_stopping_criterion(existing, criterion):
    if existing is None:
        return [criterion]
    try:
        merged = list(existing)
    except TypeError:
        merged = [existing]
    merged.append(criterion)
    return merged


def generate_until_with_optional_ttft(lm, instance, measure_ttft: bool, start_sec: float):
    if not measure_ttft:
        return lm.generate_until([instance]), {}

    model = getattr(lm, "model", None)
    original_generate = getattr(model, "generate", None)
    if model is None or original_generate is None:
        return lm.generate_until([instance]), {
            "ttft_sec": None,
            "generated_steps_observed": None,
            "ttft_error": "lm.model.generate not found",
        }

    ttft_criterion = TTFTStoppingCriteria(start_sec)

    def wrapped_generate(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["stopping_criteria"] = append_stopping_criterion(
            kwargs.get("stopping_criteria"),
            ttft_criterion,
        )
        return original_generate(*args, **kwargs)

    try:
        setattr(model, "generate", wrapped_generate)
        responses = lm.generate_until([instance])
    finally:
        setattr(model, "generate", original_generate)

    return responses, {
        "ttft_sec": ttft_criterion.first_token_sec,
        "generated_steps_observed": ttft_criterion.generated_steps,
    }


def estimate_token_counts(matching_groups_path: str, video_key: str, max_frames: int, stride: int) -> Dict[str, Any]:
    orig_grid = 27
    pooled_grid = orig_grid // stride
    tokens_per_frame = pooled_grid * pooled_grid
    total_tokens = max_frames * tokens_per_frame
    matching_id = matching_video_id(video_key)
    path = Path(matching_groups_path) / matching_id / "matching_groups.json"
    if not path.exists():
        return {
            "matching_groups_file": str(path),
            "visual_tokens_before": total_tokens,
            "visual_tokens_after_estimate": None,
            "pruned_tokens_estimate": None,
            "pruned_ratio_estimate": None,
        }

    with path.open("r") as f:
        data = json.load(f)

    num_frames = int(data.get("num_frames", max_frames))
    total_tokens = num_frames * tokens_per_frame
    keep_mask = [True] * total_tokens

    for group in data.get("groups", []):
        members = group.get("members", [])
        if len(members) < 2:
            continue
        mapped = []
        for member in members:
            frame = int(member["frame"])
            patch = int(member["patch"])
            row = patch // orig_grid
            col = patch % orig_grid
            new_row = min(int(row * pooled_grid / orig_grid), pooled_grid - 1)
            new_col = min(int(col * pooled_grid / orig_grid), pooled_grid - 1)
            idx = frame * tokens_per_frame + new_row * pooled_grid + new_col
            if 0 <= idx < total_tokens and idx not in mapped:
                mapped.append(idx)
        if len(mapped) >= 2:
            for idx in mapped[1:]:
                keep_mask[idx] = False

    kept = sum(keep_mask)
    pruned = total_tokens - kept
    return {
        "matching_groups_file": str(path),
        "visual_tokens_before": total_tokens,
        "visual_tokens_after_estimate": kept,
        "pruned_tokens_estimate": pruned,
        "pruned_ratio_estimate": pruned / total_tokens if total_tokens else None,
    }


def summarize_latencies(latencies: List[float]) -> Dict[str, Optional[float]]:
    if not latencies:
        return {"total_sec": 0.0, "mean_sec": None, "median_sec": None, "min_sec": None, "max_sec": None}
    return {
        "total_sec": sum(latencies),
        "mean_sec": statistics.mean(latencies),
        "median_sec": statistics.median(latencies),
        "min_sec": min(latencies),
        "max_sec": max(latencies),
    }


def profile_variant(
    *,
    args: argparse.Namespace,
    task,
    split: str,
    video_keys: List[str],
    grouped_doc_ids: Dict[str, List[int]],
    variant: str,
) -> Dict[str, Any]:
    if variant == "baseline":
        model_key = args.baseline_model
        model_args = baseline_model_args(args)
    elif variant == "prune":
        model_key = args.prune_model
        model_args = prune_model_args(args)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    load_start = time.perf_counter()
    lm = create_model(model_key, model_args, args.batch_size, args.device)
    lm.task_dict[args.task] = task.dataset
    cuda_sync()
    model_load_sec = time.perf_counter() - load_start

    variant_result = {
        "variant": variant,
        "model": model_key,
        "model_args": model_args,
        "model_load_sec": model_load_sec,
        "videos": [],
    }

    for video_key in video_keys:
        doc_ids = grouped_doc_ids[video_key]
        if args.max_questions is not None:
            doc_ids = doc_ids[: args.max_questions]

        warmup_ids = doc_ids[: args.warmup_questions]
        timed_ids = doc_ids[args.warmup_questions :]

        for doc_id in warmup_ids:
            instance = construct_instance(task, doc_id, split)
            _ = lm.generate_until([instance])
            cuda_sync()

        per_question = []
        latencies = []
        ttft_latencies = []
        decode_after_ttft_latencies = []
        for doc_id in timed_ids:
            doc = task.dataset[split][doc_id]
            instance = construct_instance(task, doc_id, split)
            cuda_sync()
            start = time.perf_counter()
            responses, ttft_info = generate_until_with_optional_ttft(
                lm,
                instance,
                args.measure_ttft,
                start,
            )
            cuda_sync()
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            row = {
                "doc_id": doc_id,
                "question_type": doc.get("question_type"),
                "latency_sec": elapsed,
            }
            if args.measure_ttft:
                ttft_sec = ttft_info.get("ttft_sec")
                decode_after_ttft_sec = elapsed - ttft_sec if ttft_sec is not None else None
                row["ttft_sec"] = ttft_sec
                row["decode_after_ttft_sec"] = decode_after_ttft_sec
                row["generated_steps_observed"] = ttft_info.get("generated_steps_observed")
                if "ttft_error" in ttft_info:
                    row["ttft_error"] = ttft_info["ttft_error"]
                if ttft_sec is not None:
                    ttft_latencies.append(ttft_sec)
                if decode_after_ttft_sec is not None:
                    decode_after_ttft_latencies.append(decode_after_ttft_sec)
            if args.include_responses:
                row["response"] = responses[0] if responses else None
            per_question.append(row)

        latency_summary = summarize_latencies(latencies)
        video_result = {
            "video_key": video_key,
            "matching_video_id": matching_video_id(video_key),
            "num_questions_total": len(grouped_doc_ids[video_key]),
            "num_questions_timed": len(timed_ids),
            "warmup_questions": len(warmup_ids),
            "per_question": per_question,
            "inference_total_sec": latency_summary["total_sec"],
            "inference_mean_sec": latency_summary["mean_sec"],
            "inference_median_sec": latency_summary["median_sec"],
            "inference_min_sec": latency_summary["min_sec"],
            "inference_max_sec": latency_summary["max_sec"],
        }
        if args.measure_ttft:
            ttft_summary = summarize_latencies(ttft_latencies)
            decode_after_ttft_summary = summarize_latencies(decode_after_ttft_latencies)
            video_result.update(
                {
                    "ttft_total_sec": ttft_summary["total_sec"],
                    "ttft_mean_sec": ttft_summary["mean_sec"],
                    "ttft_median_sec": ttft_summary["median_sec"],
                    "ttft_min_sec": ttft_summary["min_sec"],
                    "ttft_max_sec": ttft_summary["max_sec"],
                    "decode_after_ttft_total_sec": decode_after_ttft_summary["total_sec"],
                    "decode_after_ttft_mean_sec": decode_after_ttft_summary["mean_sec"],
                    "decode_after_ttft_median_sec": decode_after_ttft_summary["median_sec"],
                    "decode_after_ttft_min_sec": decode_after_ttft_summary["min_sec"],
                    "decode_after_ttft_max_sec": decode_after_ttft_summary["max_sec"],
                }
            )

        if variant == "baseline":
            video_result["T_baseline_video_sec"] = latency_summary["total_sec"]
        else:
            one_time = args.reconstruction_sec + args.matching_sec
            video_result["T_reconstruction_once_sec"] = args.reconstruction_sec
            video_result["T_matching_once_sec"] = args.matching_sec
            video_result["T_prune_video_sec"] = one_time + latency_summary["total_sec"]
            video_result.update(
                estimate_token_counts(
                    args.matching_groups_path,
                    video_key,
                    args.max_frames_num,
                    args.mm_spatial_pool_stride,
                )
            )

        variant_result["videos"].append(video_result)

    del lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return variant_result


def main() -> None:
    args = parse_args()
    if not args.list_videos and not args.pretrained:
        raise SystemExit("--pretrained is required unless --list-videos is used.")
    setup_import_path()

    task = load_task(args.task, args.baseline_model)
    split = get_split(task)
    groups = group_docs_by_video(task, split)

    if args.list_videos:
        rows = [
            {"video_key": key, "matching_video_id": matching_video_id(key), "num_questions": len(doc_ids)}
            for key, doc_ids in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
        ]
        print(json.dumps(rows[: args.list_limit], indent=2, ensure_ascii=False))
        return

    video_keys = select_video_keys(args, groups)
    result = {
        "task": args.task,
        "split": split,
        "selected_videos": video_keys,
        "num_selected_videos": len(video_keys),
        "measure_ttft": args.measure_ttft,
        "formula": {
            "baseline": "T_baseline_video = N * T_baseline_inference_per_question",
            "prune": "T_prune_video = T_reconstruction_once + T_matching_once + N * T_pruned_inference_per_question",
        },
        "variants": [],
    }

    variants = ["baseline", "prune"] if args.variant == "both" else [args.variant]
    for variant in variants:
        result["variants"].append(
            profile_variant(
                args=args,
                task=task,
                split=split,
                video_keys=video_keys,
                grouped_doc_ids=groups,
                variant=variant,
            )
        )

    output = json.dumps(result, indent=2, ensure_ascii=False)
    print(output)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")
        print(f"Wrote JSON to {out_path}")


if __name__ == "__main__":
    main()
