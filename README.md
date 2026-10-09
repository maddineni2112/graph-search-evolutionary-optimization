# Graph Search and Evolutionary Optimization for Automated Model Selection

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MNIST%20MLP-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-regression-013243?logo=numpy&logoColor=white)
![Project Type](https://img.shields.io/badge/Project-Fundamentals%20of%20AI-6B46C1)

A reproducible study of **automated model selection** across two supervised-learning tasks. The project converts a discrete hyperparameter space into a graph, explores it with best-first search, and refines configurations with a genetic algorithm. The same optimization interface is applied to a tabular regression task and a PyTorch image-classification task.

## Research question

Can a shared search-and-optimization controller select useful model configurations across different data modalities while keeping the model training code task-specific and reproducible?

## Contributions

1. A discrete hyperparameter graph in which each node is a complete model configuration and each edge changes one parameter.
2. A best-first graph-search baseline that evaluates promising neighboring configurations first.
3. An elitist genetic optimizer with crossover, mutation, caching, and convergence history.
4. A tabular gradient-descent regression task adapted from the original housing-price notebook.
5. A PyTorch MLP task adapted from the original MNIST notebook.
6. A single command-line runner that produces JSON experiment results.

## System overview

```text
                 Discrete hyperparameter space
                              │
                 ┌────────────┴────────────┐
                 │                         │
        Best-first graph search      Genetic optimization
                 │                         │
                 └────────────┬────────────┘
                              │
              Shared objective and result schema
                         ┌────┴────┐
                         │         │
                Housing regression  MNIST classification
                gradient descent     PyTorch MLP
```

## Tasks and models

| Task | Model | Search objective | Evaluation |
| --- | --- | --- | --- |
| Housing regression | Standardized linear regression trained with gradient descent | Holdout mean squared error | Holdout RMSE in dollars |
| MNIST classification | Fully connected PyTorch MLP with dropout | Validation loss | Test accuracy |

The housing task keeps the original eight-row coursework example as a transparent smoke-test benchmark. It is intentionally reported as pedagogical evidence, not as a statistically representative housing study. The MNIST branch downloads the dataset through `torchvision` and uses deterministic seeds.

## Repository contents

| Path | Purpose |
| --- | --- |
| [`src/evolutionary_search.py`](src/evolutionary_search.py) | Discrete search space, best-first search, and genetic optimizer |
| [`src/tasks.py`](src/tasks.py) | Housing and MNIST task adapters |
| [`src/run_project.py`](src/run_project.py) | Command-line experiment runner |
| [`src/HW01_search.ipynb`](src/HW01_search.ipynb) | Original housing-price coursework baseline |
| [`src/HW02_optimization.ipynb`](src/HW02_optimization.ipynb) | Original MNIST MLP coursework baseline |
| [`results/housing_experiment.json`](results/housing_experiment.json) | Reproduced housing search output |
| [`reports/`](reports/) | Academic report and presentation |
| [`data/README.md`](data/README.md) | Dataset provenance and access instructions |

## Reproducibility

Create an environment with Python 3.10 or newer and install the dependencies:

```bash
python -m venv .venv
```

Windows activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux activation:

```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Run the lightweight regression experiment:

```bash
python src/run_project.py --task housing
```

Run the MNIST search with a smaller training subset:

```bash
python src/run_project.py --task mnist --mnist-train-samples 10000
```

Run both tasks:

```bash
python src/run_project.py --task both
```

Results are written to `results/experiment_results.json` by default. MNIST downloads are stored locally under `data/` and are excluded from version control.

## Reproduced evidence

The housing search was executed locally with seed `42`:

| Method | Selected configuration | Holdout RMSE |
| --- | --- | ---: |
| Best-first graph search | learning rate `0.005`, `500` epochs, L2 `0.01` | `$8,959` |
| Genetic optimizer | learning rate `0.005`, `500` epochs, L2 `0.01` | `$8,959` |

The original MNIST notebook contains a saved baseline of **97.80% test accuracy after five epochs**. The integrated runner can rerun that task and record a fresh result under the environment described by `requirements.txt`.

## Limitations

- The housing benchmark contains only eight examples, so its holdout error demonstrates pipeline behavior rather than generalization to a real housing population.
- The current search space is discrete and intentionally small for reproducibility.
- The MNIST branch is compute-dependent and should be rerun before making a final comparison of search methods.
- The project does not claim that one optimizer is universally superior; it provides a common, inspectable protocol for comparing them.

## Future work

- Replace the pedagogical housing table with a documented public housing dataset.
- Add breadth-first, depth-first, and iterative-deepening traversal baselines.
- Repeat each configuration across multiple seeds and report confidence intervals.
- Add early stopping and parallel evaluation for larger search spaces.

