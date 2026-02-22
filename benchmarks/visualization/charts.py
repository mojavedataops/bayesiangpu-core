"""Benchmark visualization charts."""
import json
from pathlib import Path
from typing import List, Optional


def create_charts(results_json: str, output_dir: str = "benchmarks/results"):
    """Create benchmark comparison charts from results JSON.

    Args:
        results_json: Path to benchmark results JSON file
        output_dir: Directory to save chart images
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(results_json) as f:
        results = json.load(f)

    # Filter successful results
    successful = [r for r in results if r.get("error") is None]
    if not successful:
        print("No successful results to plot")
        return

    models = sorted(set(r["model"] for r in successful))
    frameworks = sorted(set(r["framework"] for r in successful))

    # Color palette
    colors = {
        "BayesianGPU": "#2196F3",
        "PyMC": "#4CAF50",
        "NumPyro": "#FF9800",
        "CmdStan": "#F44336",
        "brms": "#9C27B0",
    }

    # --- Wall Time Chart ---
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(models))
    width = 0.15

    for i, fw in enumerate(frameworks):
        times = []
        for model in models:
            match = [r for r in successful if r["model"] == model and r["framework"] == fw]
            times.append(match[0]["wall_time_s"] if match else 0)
        offset = (i - len(frameworks) / 2 + 0.5) * width
        bars = ax.bar(x + offset, times, width, label=fw,
                     color=colors.get(fw, "#888888"), alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Wall Time (seconds)")
    ax.set_title("Bayesian Inference: Wall Time Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(output_path / "wall_time.png", dpi=150)
    plt.close()

    # --- ESS/second Chart ---
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, fw in enumerate(frameworks):
        ess_rates = []
        for model in models:
            match = [r for r in successful if r["model"] == model and r["framework"] == fw]
            ess_rates.append(match[0]["ess_per_second"] if match else 0)
        offset = (i - len(frameworks) / 2 + 0.5) * width
        ax.bar(x + offset, ess_rates, width, label=fw,
              color=colors.get(fw, "#888888"), alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("ESS / second")
    ax.set_title("Bayesian Inference: Sampling Efficiency (ESS/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(output_path / "ess_per_second.png", dpi=150)
    plt.close()

    # --- Memory Chart ---
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, fw in enumerate(frameworks):
        memory = []
        for model in models:
            match = [r for r in successful if r["model"] == model and r["framework"] == fw]
            memory.append(match[0]["peak_memory_mb"] if match else 0)
        offset = (i - len(frameworks) / 2 + 0.5) * width
        ax.bar(x + offset, memory, width, label=fw,
              color=colors.get(fw, "#888888"), alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Bayesian Inference: Memory Usage")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path / "memory.png", dpi=150)
    plt.close()

    print(f"Charts saved to {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        create_charts(sys.argv[1])
    else:
        print("Usage: python -m benchmarks.visualization.charts results.json")
