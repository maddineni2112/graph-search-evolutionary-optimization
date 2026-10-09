"""Reusable graph-search and evolutionary optimization utilities.

The project treats a discrete hyperparameter space as a graph. Each node is
one configuration and an edge changes exactly one hyperparameter value. The
same objective interface can therefore support graph-search baselines and a
genetic optimizer across different learning tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from random import Random
from typing import Any, Callable, Iterable, Mapping


Config = dict[str, Any]
Objective = Callable[[Config], float]


@dataclass
class SearchResult:
    """Best configuration and trace returned by an optimizer."""

    method: str
    config: Config
    score: float
    history: list[float]
    evaluations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "config": self.config,
            "score": self.score,
            "history": self.history,
            "evaluations": self.evaluations,
        }


class DiscreteSearchSpace:
    """Finite Cartesian search space with deterministic neighbor expansion."""

    def __init__(self, values: Mapping[str, Iterable[Any]]) -> None:
        self.values = {name: list(options) for name, options in values.items()}
        if not self.values or any(not options for options in self.values.values()):
            raise ValueError("Every search parameter must have at least one value")
        self.names = tuple(self.values)

    def neighbors(self, config: Config) -> list[Config]:
        neighbors: list[Config] = []
        for name in self.names:
            for value in self.values[name]:
                if value != config[name]:
                    candidate = dict(config)
                    candidate[name] = value
                    neighbors.append(candidate)
        return neighbors

    def sample(self, rng: Random) -> Config:
        return {name: rng.choice(options) for name, options in self.values.items()}

    @staticmethod
    def key(config: Config) -> tuple[tuple[str, Any], ...]:
        return tuple(sorted(config.items()))


class GraphSearchOptimizer:
    """Evaluate configurations using graph traversal or best-first search."""

    def __init__(self, space: DiscreteSearchSpace, objective: Objective) -> None:
        self.space = space
        self.objective = objective

    def best_first(self, start: Config, budget: int = 20) -> SearchResult:
        """Expand the lowest-scoring frontier configuration first."""

        cache: dict[tuple[tuple[str, Any], ...], float] = {}
        counter = 0

        def evaluate(config: Config) -> float:
            nonlocal counter
            key = self.space.key(config)
            if key not in cache:
                cache[key] = float(self.objective(config))
                counter += 1
            return cache[key]

        start_score = evaluate(start)
        frontier: list[tuple[float, int, Config]] = [(start_score, 0, dict(start))]
        queued = {self.space.key(start)}
        history: list[float] = []
        best_config, best_score = dict(start), start_score

        while frontier and counter <= budget:
            score, _, current = heapq.heappop(frontier)
            if score < best_score:
                best_config, best_score = dict(current), score
            history.append(best_score)
            for candidate in self.space.neighbors(current):
                key = self.space.key(candidate)
                if key in queued or counter >= budget:
                    continue
                candidate_score = evaluate(candidate)
                heapq.heappush(frontier, (candidate_score, len(queued), candidate))
                queued.add(key)

        return SearchResult("best_first", best_config, best_score, history, counter)


class GeneticOptimizer:
    """Simple elitist genetic optimizer for a finite categorical search space."""

    def __init__(
        self,
        space: DiscreteSearchSpace,
        objective: Objective,
        seed: int = 42,
    ) -> None:
        self.space = space
        self.objective = objective
        self.rng = Random(seed)

    def _crossover(self, left: Config, right: Config) -> Config:
        child = {}
        for name in self.space.names:
            child[name] = left[name] if self.rng.random() < 0.5 else right[name]
        return child

    def _mutate(self, config: Config, mutation_rate: float) -> Config:
        child = dict(config)
        for name in self.space.names:
            if self.rng.random() < mutation_rate:
                child[name] = self.rng.choice(self.space.values[name])
        return child

    def run(
        self,
        population_size: int = 8,
        generations: int = 6,
        elite_fraction: float = 0.25,
        mutation_rate: float = 0.15,
    ) -> SearchResult:
        population = [self.space.sample(self.rng) for _ in range(population_size)]
        cache: dict[tuple[tuple[str, Any], ...], float] = {}

        def evaluate(config: Config) -> float:
            key = self.space.key(config)
            if key not in cache:
                cache[key] = float(self.objective(config))
            return cache[key]

        best_config: Config | None = None
        best_score = float("inf")
        history: list[float] = []

        for _ in range(generations):
            ranked = sorted(((evaluate(config), config) for config in population), key=lambda item: item[0])
            if ranked[0][0] < best_score:
                best_score, best_config = ranked[0][0], dict(ranked[0][1])
            history.append(best_score)
            elite_count = max(1, int(population_size * elite_fraction))
            elites = [config for _, config in ranked[:elite_count]]
            next_population = [dict(config) for config in elites]
            while len(next_population) < population_size:
                left = self.rng.choice(elites)
                right = self.rng.choice(elites)
                next_population.append(self._mutate(self._crossover(left, right), mutation_rate))
            population = next_population

        assert best_config is not None
        return SearchResult("genetic", best_config, best_score, history, len(cache))
