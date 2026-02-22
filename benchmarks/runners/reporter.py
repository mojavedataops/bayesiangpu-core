"""Benchmark result reporting."""
import json
from typing import List
from ..adapters.base import BenchmarkResult


def to_markdown_table(results: List[BenchmarkResult]) -> str:
    """Generate a Markdown comparison table."""
    # Group by model
    models = sorted(set(r.model_name for r in results))
    frameworks = sorted(set(r.framework for r in results))

    lines = ["# Benchmark Results\n"]

    # Wall time table
    lines.append("## Wall Time (seconds)\n")
    header = "| Model | " + " | ".join(frameworks) + " |"
    sep = "|" + "---|" * (len(frameworks) + 1)
    lines.extend([header, sep])

    for model in models:
        row = f"| {model} |"
        for fw in frameworks:
            match = [r for r in results if r.model_name == model and r.framework == fw]
            if match and match[0].succeeded:
                row += f" {match[0].wall_time_seconds:.2f} |"
            elif match:
                row += f" {match[0].error} |"
            else:
                row += " N/A |"
        lines.append(row)

    # ESS/second table
    lines.append("\n## ESS per Second\n")
    lines.extend([header, sep])

    for model in models:
        row = f"| {model} |"
        for fw in frameworks:
            match = [r for r in results if r.model_name == model and r.framework == fw]
            if match and match[0].succeeded:
                row += f" {match[0].ess_per_second:.1f} |"
            elif match:
                row += f" - |"
            else:
                row += " N/A |"
        lines.append(row)

    return "\n".join(lines)


def to_json(results: List[BenchmarkResult], path: str):
    """Save results to JSON."""
    data = []
    for r in results:
        data.append({
            "framework": r.framework,
            "model": r.model_name,
            "wall_time_s": r.wall_time_seconds,
            "peak_memory_mb": r.peak_memory_mb,
            "ess_per_second": r.ess_per_second,
            "min_ess": r.min_ess,
            "error": r.error,
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
