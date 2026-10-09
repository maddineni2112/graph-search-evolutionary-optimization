"""Run the integrated graph-search and evolutionary optimization study.

Examples:
    python src/run_project.py --task housing
    python src/run_project.py --task mnist --mnist-train-samples 10000
    python src/run_project.py --task both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evolutionary_search import DiscreteSearchSpace, GeneticOptimizer, GraphSearchOptimizer
from tasks import HousingTask, MNISTTask


def run_housing() -> dict[str, Any]:
    task = HousingTask()
    space = DiscreteSearchSpace(
        {
            "learning_rate": [0.005, 0.01, 0.02],
            "epochs": [500, 1000, 1500],
            "l2": [0.0, 0.001, 0.01],
        }
    )
    start = {"learning_rate": 0.01, "epochs": 1000, "l2": 0.0}
    graph_result = GraphSearchOptimizer(space, task.evaluate).best_first(start, budget=18)
    genetic_result = GeneticOptimizer(space, task.evaluate, seed=42).run(
        population_size=8, generations=5, mutation_rate=0.2
    )
    best = genetic_result if genetic_result.score <= graph_result.score else graph_result
    return {
        "task": "housing_regression",
        "baseline_from_original_notebook": {"query_prediction_dollars": 425005.21},
        "best_first": graph_result.to_dict(),
        "genetic": genetic_result.to_dict(),
        "selected": task.summary(best.config),
    }


def run_mnist(train_samples: int) -> dict[str, Any]:
    task = MNISTTask()
    space = DiscreteSearchSpace(
        {
            "hidden1": [128, 256],
            "hidden2": [64, 128],
            "dropout": [0.2, 0.3, 0.4],
            "learning_rate": [0.0005, 0.001],
            "epochs": [3, 5],
            "train_samples": [train_samples],
        }
    )
    result = GeneticOptimizer(space, task.evaluate, seed=42).run(
        population_size=6, generations=4, mutation_rate=0.2
    )
    validation_loss, test_accuracy = task.run(result.config, train_samples=train_samples)
    return {
        "task": "mnist_image_classification",
        "baseline_from_original_notebook": {"test_accuracy": 0.978, "epochs": 5},
        "genetic": result.to_dict(),
        "selected": {
            "config": result.config,
            "validation_loss": validation_loss,
            "test_accuracy": test_accuracy,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["housing", "mnist", "both"], default="both")
    parser.add_argument("--mnist-train-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("results/experiment_results.json"))
    args = parser.parse_args()

    results: dict[str, Any] = {
        "project": "Evolutionary Model Selection for Regression and Image Classification",
        "seed": 42,
    }
    if args.task in {"housing", "both"}:
        results["housing"] = run_housing()
    if args.task in {"mnist", "both"}:
        results["mnist"] = run_mnist(args.mnist_train_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
