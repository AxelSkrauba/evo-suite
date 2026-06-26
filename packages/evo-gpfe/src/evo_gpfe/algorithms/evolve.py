"""Evolutionary loop that breeds a single engineered feature."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from deap import base, tools

from evo_gpfe.core.config import GPConfig


def evolve_one_feature(
    toolbox: base.Toolbox,
    config: GPConfig,
    prev_population: list | None = None,
) -> tuple[Any, int, list]:
    """Evolve the population to produce one optimal feature tree.

    Applies tournament selection, subtree crossover, mixed mutation
    (subtree / hoist / point), elitism and stagnation-based early stopping.

    Parameters
    ----------
    toolbox : deap.base.Toolbox
        Configured toolbox with ``population``, ``evaluate``, ``select``,
        ``mate``, ``mutate_subtree``, ``mutate_hoist``, ``mutate_point`` and
        ``clone``.
    config : GPConfig
        Algorithm configuration.
    prev_population : list, optional
        Population from the previous feature, reused (warm start) with fitness
        invalidated because the hall of fame changed.

    Returns
    -------
    best_tree : object
        The best individual found.
    best_gen : int
        Generation at which it was found.
    population : list
        Final population (for the next warm start).
    """
    p_subtree, p_hoist, _ = config.mutation_split()

    if prev_population is not None and config.warm_start_population:
        population = [toolbox.clone(ind) for ind in prev_population]
        for ind in population:
            del ind.fitness.values
    else:
        population = toolbox.population(n=config.population_size)

    for ind, fit in zip(population, map(toolbox.evaluate, population)):
        ind.fitness.values = fit

    best_ever = toolbox.clone(tools.selBest(population, 1)[0])
    best_gen = 0
    stagnation = 0

    for gen in range(config.n_generations):
        elites = [toolbox.clone(e) for e in tools.selBest(population, config.elite_size)]

        offspring = [toolbox.clone(ind) for ind in toolbox.select(population, len(population))]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < config.crossover_prob:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        for mutant in offspring:
            if random.random() < config.mutation_prob:
                r = random.random()
                if r < p_subtree:
                    toolbox.mutate_subtree(mutant)
                elif r < p_subtree + p_hoist:
                    toolbox.mutate_hoist(mutant)
                else:
                    toolbox.mutate_point(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        offspring.sort(key=lambda ind: ind.fitness.values[0])
        for i, elite in enumerate(elites):
            offspring[i] = elite
        population[:] = offspring

        gen_best = tools.selBest(population, 1)[0]
        if gen_best.fitness.values[0] > best_ever.fitness.values[0]:
            best_ever = toolbox.clone(gen_best)
            best_gen = gen
            stagnation = 0
        else:
            stagnation += 1

        if config.verbose and (gen % 10 == 0 or gen == config.n_generations - 1):
            fits = [ind.fitness.values[0] for ind in population]
            sizes = [len(ind) for ind in population]
            print(
                f"    Gen {gen:3d} | best={max(fits):.4f} "
                f"| mean={np.mean(fits):.4f} | mean_nodes={np.mean(sizes):.1f}"
            )

        if stagnation >= config.early_stopping_rounds:
            if config.verbose:
                print(f"    -> early stopping at gen {gen} (stagnation)")
            break

    return best_ever, best_gen, population
