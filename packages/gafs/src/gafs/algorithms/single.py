"""Single-objective genetic algorithm with elitism and early stopping."""

from __future__ import annotations

import random
import time
from typing import Callable

import numpy as np
from deap import base, tools

from gafs.core.config import EvolutionStats, GAConfig

Callback = Callable[[int, EvolutionStats, list], bool]


def _n_features_of(individual: list[int]) -> int:
    return int(sum(individual))


def run_single_objective(
    toolbox: base.Toolbox,
    config: GAConfig,
    callbacks: list[Callback] | None = None,
) -> tuple[list, tools.Logbook, list[EvolutionStats]]:
    """Run the single-objective GA.

    The loop applies tournament selection, uniform crossover, bit-flip mutation
    and elitism (the best ``elite_size`` individuals survive unchanged). It
    supports both configuration-driven early stopping and callback-driven
    stopping (a callback returning ``True`` halts evolution).

    Parameters
    ----------
    toolbox : deap.base.Toolbox
        Pre-configured toolbox with ``population``, ``evaluate``, ``select``,
        ``mate``, ``mutate`` and ``clone`` registered.
    config : GAConfig
        Algorithm configuration.
    callbacks : list of callable, optional
        Functions ``f(gen, stats, population) -> bool``.

    Returns
    -------
    population : list
        Final population.
    logbook : deap.tools.Logbook
        DEAP logbook of compiled statistics.
    history : list of EvolutionStats
        Per-generation statistics.
    """
    population = toolbox.population(n=config.population_size)
    logbook = tools.Logbook()
    history: list[EvolutionStats] = []

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("mean", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    for ind, fit in zip(population, map(toolbox.evaluate, population)):
        ind.fitness.values = fit

    best_fitness_history: list[float] = []
    no_improve_count = 0

    for gen in range(config.n_generations):
        t_gen_start = time.time()

        elites = [toolbox.clone(e) for e in tools.selBest(population, config.elite_size)]

        offspring = [toolbox.clone(ind) for ind in toolbox.select(population, len(population))]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < config.crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < config.mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        # Re-insert elites in place of the worst offspring.
        offspring.sort(key=lambda ind: ind.fitness.values[0])
        for i, elite in enumerate(elites):
            offspring[i] = elite
        population[:] = offspring

        fits = [ind.fitness.values[0] for ind in population]
        n_feats = [_n_features_of(ind) for ind in population]
        best_ind = tools.selBest(population, 1)[0]

        gen_stats = EvolutionStats(
            generation=gen,
            best_fitness=float(max(fits)),
            mean_fitness=float(np.mean(fits)),
            std_fitness=float(np.std(fits)),
            best_n_features=_n_features_of(best_ind),
            mean_n_features=float(np.mean(n_feats)),
            elapsed_time=time.time() - t_gen_start,
        )
        history.append(gen_stats)
        logbook.record(gen=gen, **stats.compile(population))

        if config.verbose and (gen % 10 == 0 or gen == config.n_generations - 1):
            print(repr(gen_stats))

        if callbacks and any(cb(gen, gen_stats, population) for cb in callbacks):
            if config.verbose:
                print(f"  Stopped by callback at generation {gen}.")
            break

        if config.early_stopping_rounds is not None:
            best_fitness_history.append(float(max(fits)))
            if len(best_fitness_history) > config.early_stopping_rounds:
                window = best_fitness_history[-config.early_stopping_rounds :]
                if max(window) - min(window) < config.early_stopping_tol:
                    no_improve_count += 1
                    if no_improve_count >= config.early_stopping_rounds:
                        if config.verbose:
                            print(f"  Early stopping at generation {gen}.")
                        break
                else:
                    no_improve_count = 0

    return population, logbook, history
