# Data and reproducibility statement

## Housing regression

The housing branch uses the eight-row feature table embedded in `src/tasks.py`, adapted from the original coursework notebook. It contains location, size, bedrooms, bathrooms, garage, age, and price fields. No external housing file is required for the smoke-test experiment.

Because this table is intentionally small and instructional, the reported holdout error is not evidence of performance on a real housing population. Future work should replace it with a documented public dataset and an approved train/validation/test protocol.

## MNIST image classification

The image branch uses the MNIST handwritten-digit dataset through `torchvision.datasets.MNIST`.

- **Origin:** Yann LeCun, Corinna Cortes, and Christopher J.C. Burges, MNIST database of handwritten digits.
- **Source:** <http://yann.lecun.com/exdb/mnist/>
- **Runtime path:** `data/MNIST/` or the path configured by `torchvision`.
- **Access:** run `python src/run_project.py --task mnist`; the dataset downloads only when it is not already cached.
- **Exclusion:** downloaded raw files and caches are not committed to this repository.

## Excluded course materials

Original course-provided files are not redistributed here. Equivalent small inputs are embedded only where needed to make the algorithmic smoke test reproducible. Keep any instructional or licensed source files outside the repository and document their schema before using them.
