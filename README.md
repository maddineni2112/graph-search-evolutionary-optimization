# Search-Guided Evolutionary Optimization for Graph Pathfinding

This course project compares classical graph search with evolutionary path optimization on reproducibly generated obstacle grids. It evaluates BFS, DFS, Iterative Deepening, Uniform-Cost Search, Greedy Best-First Search, and A*, then tests whether seeding a Genetic Algorithm with an A* route improves convergence and reliability.

The primary implementation is the executed notebook src/Graph_Search_Evolutionary_Optimization.ipynb. The original HW01 and HW02 notebooks are preserved under legacy/ as coursework provenance. They are documented honestly: HW01 is housing-price regression and HW02 is an MNIST MLP, not graph search.

## Problem statement

Given a start node, goal node, obstacles, and optional terrain costs, find a feasible route while balancing path cost, path length, search effort, runtime, and stochastic stability.

## Research questions

1. How do uninformed, cost-based, and heuristic searches compare on unit-cost and weighted grids?
2. Does A* seeding improve Genetic Algorithm success rate, final fitness, convergence, or variability?
3. How do grid size and obstacle density affect search effort and evolutionary runtime?

## Methods

### Classical search

- Breadth-First Search
- Depth-First Search
- Iterative Deepening Search
- Uniform-Cost Search/Dijkstra
- Greedy Best-First Search
- A*

### Evolutionary search

The Genetic Algorithm represents a route as a fixed-length sequence of four-direction actions. The fitness function penalizes incomplete routes, invalid moves, revisits, and expensive paths. The hybrid variant inserts an A* route into the initial population and mutates additional variants around it.

## Experimental setup

The notebook generates 12 deterministic scenarios:

- grid sizes: 15x15, 25x25, 35x35;
- obstacle densities: 15% and 25%;
- unit-cost and weighted-terrain modes;
- weighted terrain costs: {1, 3, 7};
- GA seeds: 11, 22, 33, 44, 55;
- population size: 30;
- generations: 50;
- elite count: 2;
- crossover probability: 0.8;
- mutation probability: 0.05.

The classical comparison contains 72 rows, the evolutionary trial file contains 120 rows, and the convergence file contains 6,000 generation-level rows.

## Verified result summary

These values come from the executed notebook and the committed CSV files.

| Method | Unit maps success | Unit mean path cost | Weighted maps success | Weighted mean path cost |
|---|---:|---:|---:|---:|
| BFS | 100% | 48.33 | 100% | 184.67 |
| DFS | 100% | 73.67 | 100% | 229.67 |
| IDS | 100% | 48.33 | 100% | 184.67 |
| UCS | 100% | 48.33 | 100% | 122.00 |
| Greedy | 100% | 53.33 | 100% | 153.33 |
| A* | 100% | 48.33 | 100% | 122.00 |
| Random GA | 70.0% | 78.48* | 76.7% | 261.48* |
| A*-seeded GA | 100% | 48.33 | 100% | 122.00 |

* GA mean path cost uses successful trials only. Final fitness and variability for every trial are available in results/stochastic_trials.csv.

Across all 120 evolutionary trials, random initialization reached the goal in 73.3% of trials, while A*-seeded initialization reached it in 100%. Mean final fitness was 565.746 for random GA and 87.658 for A*-seeded GA. These are measured outcomes for this benchmark, not universal claims about all pathfinding problems.

## Visual evidence

- Representative paths: results/figures/representative_paths.svg
- Classical search comparison: results/figures/classical_search_comparison.svg
- GA convergence: results/figures/ga_convergence.svg

## Repository structure

~~~~
.
├── README.md
├── data/
│   └── README.md
├── legacy/
│   ├── HW01_search.ipynb
│   ├── HW02_optimization.ipynb
│   └── README.md
├── results/
│   ├── algorithm_comparison.csv
│   ├── stochastic_trials.csv
│   ├── convergence.csv
│   ├── experiment_manifest.json
│   └── figures/
├── reports/
│   ├── Graph_Search_Evolutionary_Optimization_Report.pdf
│   └── Graph_Search_Evolutionary_Optimization_Presentation.pptx
├── src/
│   └── Graph_Search_Evolutionary_Optimization.ipynb
└── requirements.txt
~~~~

## Installation and execution

~~~~
python -m venv .venv
~~~~

Windows:

~~~~
.venv/Scripts/activate
~~~~

macOS/Linux:

~~~~
source .venv/bin/activate
~~~~

Install the notebook tooling:

~~~~
pip install -r requirements.txt
~~~~

Execute the complete project notebook:

~~~~
jupyter nbconvert --to notebook --execute --inplace src/Graph_Search_Evolutionary_Optimization.ipynb
~~~~

Run it from the repository root or from the src/ directory. The notebook regenerates results/ and results/figures/.

## Report and presentation

- Academic report: reports/Graph_Search_Evolutionary_Optimization_Report.pdf
- Project presentation: reports/Graph_Search_Evolutionary_Optimization_Presentation.pptx

## Reproducibility and limitations

The benchmark is synthetic and deterministic, with fixed scenario and GA seeds. Runtime values depend on the execution environment. The experiment uses small-to-medium grid sizes and a simple action-based chromosome, so the results do not establish universal algorithm rankings. The A*-seeded method has access to a classical solution by design, and its initialization cost is recorded separately.

## References

1. E. W. Dijkstra, “A note on two problems in connexion with graphs,” Numerische Mathematik, vol. 1, pp. 269–271, 1959. doi: https://doi.org/10.1007/BF01386390.
2. P. E. Hart, N. J. Nilsson, and B. Raphael, “A formal basis for the heuristic determination of minimum cost paths,” IEEE Transactions on Systems Science and Cybernetics, vol. 4, no. 2, pp. 100–107, 1968. doi: https://doi.org/10.1109/TSSC.1968.300136.
3. J. H. Holland, Adaptation in Natural and Artificial Systems. University of Michigan Press, 1975.
