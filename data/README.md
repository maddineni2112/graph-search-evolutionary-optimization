# Benchmark and data provenance

The primary project does not require a downloaded or restricted dataset. It generates deterministic four-neighbor obstacle grids inside the executed notebook.

## Generation procedure

- Grid sizes: 15x15, 25x25, and 35x35.
- Obstacle densities: 0.15 and 0.25.
- Modes: unit-cost and weighted terrain.
- Weighted terrain costs: {1, 3, 7} for traversable cells.
- Scenario seeds: 7000 through 7011.
- GA seeds: 11, 22, 33, 44, and 55.
- The generator reserves the start and goal cells and retries with a deterministic seed offset until a route exists.

The exact generated scenario metadata is stored in results/experiment_manifest.json. The notebook regenerates the same benchmark and writes the machine-readable result files.

## Legacy coursework data

The original HW01 notebook used an embedded eight-row housing-price example. The original HW02 notebook downloaded MNIST at runtime. Those materials remain only as historical coursework under legacy/; they are not used by the graph-pathfinding experiment and no raw course files or downloaded datasets are committed.
