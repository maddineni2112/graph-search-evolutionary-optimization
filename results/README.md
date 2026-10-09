# Experiment results

`run_project.py` writes JSON result files here. The checked-in `housing_experiment.json` is the locally reproduced NumPy-only run using seed `42`.

MNIST results are environment-dependent because the task downloads data and trains a PyTorch model. Generate them with:

```bash
python src/run_project.py --task mnist --mnist-train-samples 10000
```

Each result records the search method, selected configuration, objective score, evaluation count, and convergence history.
