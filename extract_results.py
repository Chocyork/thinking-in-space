#!/usr/bin/env python3
"""Extract all VSI-Bench results from logs directory."""

import json
import os
from pathlib import Path
from collections import defaultdict
import datetime

LOGS_DIR = Path("/home/qyk/thinking-in-space/logs")

# Map log directory names to experiment configs
# Format: (model_size, strategy, matching_data, alpha)
EXPERIMENTS = {
    # Baselines
    "vsibench_baseline_7b_16f": {"model": "7B", "strategy": "baseline", "matching": "none", "alpha": None, "pruned_pct": 0.0},
    "baseline_16f": {"model": "7B", "strategy": "baseline", "matching": "none", "alpha": None, "pruned_pct": 0.0},
    "vsibench_llava_video_7b_baseline_16f": {"model": "7B-Video", "strategy": "baseline", "matching": "none", "alpha": None, "pruned_pct": 0.0},
    
    # 3D Smooth (voxel10cm)
    "vsibench_3d_smooth_alpha0.3": {"model": "0.5B", "strategy": "smooth", "matching": "voxel10cm", "alpha": 0.3, "pruned_pct": 0.0},
    "vsibench_3d_smooth_alpha0.4": {"model": "0.5B", "strategy": "smooth", "matching": "voxel10cm", "alpha": 0.4, "pruned_pct": 0.0},
    "vsibench_3d_smooth_alpha0.5": {"model": "0.5B", "strategy": "smooth", "matching": "voxel10cm", "alpha": 0.5, "pruned_pct": 0.0},
    "vsibench_7b_smooth_alpha0.3": {"model": "7B", "strategy": "smooth", "matching": "voxel10cm", "alpha": 0.3, "pruned_pct": 0.0},
    
    # 3D Prune (keep first)
    "vsibench_3d_prune_alpha0.3": {"model": "0.5B", "strategy": "prune_first", "matching": "voxel10cm", "alpha": 0.3, "pruned_pct": 66.5},
    "vsibench_7b_prune_alpha0.3": {"model": "7B", "strategy": "prune_first", "matching": "voxel10cm", "alpha": 0.3, "pruned_pct": 66.5},
    
    # 3D Prune Max
    "vsibench_0.5b_prune_max": {"model": "0.5B", "strategy": "prune_max", "matching": "voxel10cm", "alpha": 0.4, "pruned_pct": 66.5},
    "vsibench_7b_prune_max": {"model": "7B", "strategy": "prune_max", "matching": "voxel10cm", "alpha": 0.4, "pruned_pct": 66.5},
    
    # 3D Smooth Pruned
    "vsibench_3d_smooth_pruned_alpha0.3": {"model": "0.5B", "strategy": "smooth_pruned", "matching": "voxel10cm", "alpha": 0.3, "pruned_pct": 66.5},
    "vsibench_3d_smooth_pruned_alpha0.5": {"model": "0.5B", "strategy": "smooth_pruned", "matching": "voxel10cm", "alpha": 0.5, "pruned_pct": 66.5},
    "vsibench_7b_smooth_pruned_alpha0.3": {"model": "7B", "strategy": "smooth_pruned", "matching": "voxel10cm", "alpha": 0.3, "pruned_pct": 66.5},
    
    # DA3 + smooth_max_v2
    "vsibench_da3_smooth_max_v2_0.3": {"model": "0.5B", "strategy": "smooth_max_v2", "matching": "da3_2dir", "alpha": 0.3, "pruned_pct": 20.0},
    "vsibench_da3_smooth_max_v2_0.4": {"model": "0.5B", "strategy": "smooth_max_v2", "matching": "da3_2dir", "alpha": 0.4, "pruned_pct": 20.0},
    "vsibench_da3_smooth_max_v2_7b_0.3": {"model": "7B", "strategy": "smooth_max_v2", "matching": "da3_2dir", "alpha": 0.3, "pruned_pct": 20.0},
    "vsibench_da3_smooth_max_v2_7b_0.4": {"model": "7B", "strategy": "smooth_max_v2", "matching": "da3_2dir", "alpha": 0.4, "pruned_pct": 20.0},
    
    # LLaVA-Video
    "vsibench_llava_video_7b_smooth_alpha0.3": {"model": "7B-Video", "strategy": "smooth", "matching": "voxel30cm", "alpha": 0.3, "pruned_pct": 97.9},
}


def parse_timestamp(dirname):
    """Parse timestamp from lmms-eval subdir like 0406_1229_..."""
    try:
        date_str = dirname[:9]  # MMDD_HHMM
        dt = datetime.datetime.strptime(date_str, "%m%d_%H%M")
        return dt
    except:
        return datetime.datetime.min


def extract_score(results_path):
    """Extract overall vsibench score from results.json."""
    try:
        with open(results_path) as f:
            data = json.load(f)
        return data["results"]["vsibench"]["vsibench_score,none"]["overall"]
    except Exception as e:
        return None


def main():
    results = []
    
    for log_dir_name, config in EXPERIMENTS.items():
        log_dir = LOGS_DIR / log_dir_name
        if not log_dir.exists():
            continue
        
        # Find all results.json under this log dir
        result_files = list(log_dir.rglob("results.json"))
        if not result_files:
            continue
        
        # Pick the latest run if multiple
        result_files.sort(key=lambda p: parse_timestamp(p.parent.name), reverse=True)
        
        scores = []
        for rf in result_files:
            score = extract_score(rf)
            if score is not None:
                scores.append((score, rf.parent.name))
        
        if not scores:
            continue
        
        # Use latest score
        best_score, subdir_name = scores[0]
        
        # If multiple runs, also show average
        avg_score = sum(s[0] for s in scores) / len(scores)
        
        results.append({
            "log_dir": log_dir_name,
            "model": config["model"],
            "strategy": config["strategy"],
            "matching": config["matching"],
            "alpha": config["alpha"],
            "pruned_pct": config["pruned_pct"],
            "score": best_score,
            "avg_score": avg_score,
            "num_runs": len(scores),
            "latest_subdir": subdir_name,
        })
    
    # Sort by model, then by score
    results.sort(key=lambda x: (x["model"], -x["score"]))
    
    print(f"{'Log Dir':<45} {'Model':<10} {'Strategy':<16} {'Matching':<12} {'Alpha':<6} {'Pruned%':<8} {'Score':<8} {'Runs':<5}")
    print("-" * 120)
    for r in results:
        alpha_str = str(r["alpha"]) if r["alpha"] is not None else "N/A"
        print(f"{r['log_dir']:<45} {r['model']:<10} {r['strategy']:<16} {r['matching']:<12} {alpha_str:<6} {r['pruned_pct']:<8.1f} {r['score']:<8.3f} {r['num_runs']:<5}")
    
    # Save to JSON for plotting
    with open(LOGS_DIR / "../extracted_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved {len(results)} results to extracted_results.json")
    return results


if __name__ == "__main__":
    main()
