#!/usr/bin/env python3
"""
Plot Accuracy-Cost Pareto curves for different pruning strategies.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_FILE = Path("/home/qyk/thinking-in-space/extracted_results.json")
OUTPUT_DIR = Path("/home/qyk/thinking-in-space")

# Color and marker mapping for strategies
STYLE_MAP = {
    "baseline":       {"color": "#2E86AB", "marker": "o", "label": "Baseline (no prune)"},
    "smooth":         {"color": "#A23B72", "marker": "s", "label": "3D Smooth (no prune)"},
    "prune_first":    {"color": "#F18F01", "marker": "^", "label": "3D Prune (keep first)"},
    "prune_max":      {"color": "#C73E1D", "marker": "D", "label": "3D Prune Max (L2)"},
    "smooth_pruned":  {"color": "#6A4C93", "marker": "v", "label": "3D Smooth+Pruned"},
    "smooth_max_v2":  {"color": "#1B998B", "marker": "p", "label": "3D Smooth+MaxPrune v2"},
}

MATCHING_LABELS = {
    "none": "No matching",
    "voxel10cm": "voxel10cm (66.5% prune)",
    "voxel30cm": "voxel30cm (97.9% prune)",
    "da3_2dir": "da3_2dir (20.0% prune)",
}


def plot_pareto(results, model_filter, title, filename):
    """Plot Pareto curve for a specific model size."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Filter results
    filtered = [r for r in results if r["model"] == model_filter]
    
    # Group by strategy for legend
    strategies_plotted = set()
    
    for r in filtered:
        strategy = r["strategy"]
        style = STYLE_MAP.get(strategy, {"color": "gray", "marker": "x", "label": strategy})
        
        # X-axis: remaining token percentage
        remain_pct = 100.0 - r["pruned_pct"]
        score = r["score"]
        
        # Skip outliers (score < 15 likely indicates run failure)
        if score < 15:
            continue
        
        label = style["label"] if strategy not in strategies_plotted else None
        strategies_plotted.add(strategy)
        
        ax.scatter(
            remain_pct, score,
            color=style["color"],
            marker=style["marker"],
            s=200,
            alpha=0.85,
            edgecolors='white',
            linewidths=1.5,
            label=label,
            zorder=5,
        )
        
        # Annotate with matching type if not baseline
        if strategy != "baseline":
            annot = f"{r['matching']}"
            if r["alpha"] is not None:
                annot += f"\nα={r['alpha']}"
            ax.annotate(
                annot,
                (remain_pct, score),
                textcoords="offset points",
                xytext=(10, 5),
                fontsize=7,
                alpha=0.8,
            )
    
    # Draw baseline horizontal line
    baseline_scores = [r["score"] for r in filtered if r["strategy"] == "baseline" and r["score"] > 15]
    if baseline_scores:
        baseline_avg = np.mean(baseline_scores)
        ax.axhline(y=baseline_avg, color="#2E86AB", linestyle="--", alpha=0.4, linewidth=1)
        ax.text(
            ax.get_xlim()[1] * 0.95, baseline_avg + 0.3,
            f"Baseline ≈ {baseline_avg:.1f}",
            ha="right", va="bottom", fontsize=9, color="#2E86AB", alpha=0.7
        )
    
    ax.set_xlabel("Remaining Visual Tokens (% of 2704)", fontsize=12)
    ax.set_ylabel("VSI-Bench Overall Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xlim(0, 105)
    
    # Add annotation explaining left = faster
    ax.text(
        0.02, 0.02, "← Faster (more pruning)",
        transform=ax.transAxes, fontsize=9, color="gray", alpha=0.7,
        ha="left", va="bottom"
    )
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_combined_pareto(results):
    """Plot all models on one figure with subplots."""
    models = ["0.5B", "7B", "7B-Video"]
    titles = {
        "0.5B": "LLaVA-OneVision 0.5B",
        "7B": "LLaVA-OneVision 7B",
        "7B-Video": "LLaVA-Video 7B",
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        filtered = [r for r in results if r["model"] == model]
        strategies_plotted = set()
        
        for r in filtered:
            strategy = r["strategy"]
            style = STYLE_MAP.get(strategy, {"color": "gray", "marker": "x", "label": strategy})
            remain_pct = 100.0 - r["pruned_pct"]
            score = r["score"]
            
            if score < 15:
                continue
            
            label = style["label"] if strategy not in strategies_plotted else None
            strategies_plotted.add(strategy)
            
            ax.scatter(
                remain_pct, score,
                color=style["color"],
                marker=style["marker"],
                s=200,
                alpha=0.85,
                edgecolors='white',
                linewidths=1.5,
                label=label,
                zorder=5,
            )
            
            if strategy != "baseline":
                annot = f"{r['matching']}"
                if r["alpha"] is not None:
                    annot += f" α={r['alpha']}"
                ax.annotate(annot, (remain_pct, score), textcoords="offset points",
                           xytext=(8, 5), fontsize=6, alpha=0.75)
        
        # Baseline line
        baseline_scores = [r["score"] for r in filtered if r["strategy"] == "baseline" and r["score"] > 15]
        if baseline_scores:
            baseline_avg = np.mean(baseline_scores)
            ax.axhline(y=baseline_avg, color="#2E86AB", linestyle="--", alpha=0.4, linewidth=1)
        
        ax.set_xlabel("Remaining Tokens (%)", fontsize=11)
        ax.set_ylabel("VSI-Bench Score", fontsize=11)
        ax.set_title(titles.get(model, model), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim(0, 105)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / "pareto_all_models.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def print_pareto_table(results):
    """Print a sorted Pareto table for each model."""
    print("\n" + "="*80)
    print("PARETO ANALYSIS: Best score for each pruning level")
    print("="*80)
    
    for model in ["0.5B", "7B", "7B-Video"]:
        filtered = [r for r in results if r["model"] == model and r["score"] > 15]
        if not filtered:
            continue
        
        print(f"\n{model}:")
        print(f"{'Remain%':>10} {'Pruned%':>10} {'Score':>8} {'Strategy':<20} {'Matching':<12} {'Alpha':>6}")
        print("-" * 70)
        
        # Sort by remaining tokens (ascending = more pruning)
        filtered.sort(key=lambda x: x["pruned_pct"], reverse=True)
        
        for r in filtered:
            remain = 100.0 - r["pruned_pct"]
            alpha_str = str(r["alpha"]) if r["alpha"] is not None else "-"
            print(f"{remain:>9.1f}% {r['pruned_pct']:>9.1f}% {r['score']:>8.2f} {r['strategy']:<20} {r['matching']:<12} {alpha_str:>6}")
        
        # Identify Pareto-optimal points
        print("\n  Pareto-optimal (best score at each pruning level or better):")
        sorted_by_prune = sorted(filtered, key=lambda x: x["pruned_pct"])
        best_score_so_far = -1
        pareto = []
        for r in sorted_by_prune:
            if r["score"] > best_score_so_far:
                pareto.append(r)
                best_score_so_far = r["score"]
        
        for r in pareto:
            remain = 100.0 - r["pruned_pct"]
            marker = "★" if r["strategy"] != "baseline" else ""
            print(f"    {marker} {remain:.1f}% remain → {r['score']:.2f} ({r['strategy']})")


def main():
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    
    # Individual plots
    plot_pareto(results, "0.5B", "Pareto Curve: LLaVA-OneVision 0.5B", "pareto_0.5b.png")
    plot_pareto(results, "7B", "Pareto Curve: LLaVA-OneVision 7B", "pareto_7b.png")
    plot_pareto(results, "7B-Video", "Pareto Curve: LLaVA-Video 7B", "pareto_7b_video.png")
    
    # Combined
    plot_combined_pareto(results)
    
    # Table
    print_pareto_table(results)


if __name__ == "__main__":
    main()
