"""Multi-objective genetic algorithm: NSGA-II."""

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


def run_nsga2(
    toolbox: base.Toolbox,
    config: GAConfig,
    callbacks: list[Callback] | None = None,
) -> tuple[list, tools.Logbook, list[EvolutionStats]]:
    """Run NSGA-II for two maximised objectives: ``cv_score`` and ``compression``.

    NSGA-II is the canonical multi-objective evolutionary algorithm: it ranks
    the population by Pareto dominance and crowding distance to preserve a
    diverse front.

    Notes
    -----
    ``selTournamentDCD`` requires the selection size to be a multiple of four.
    We round the population size up to the next multiple of four and pad with
    clones, then trim the offspring back to ``population_size``.

    Parameters
    ----------
    toolbox : deap.base.Toolbox
        Pre-configured toolbox (``select`` must be ``selNSGA2``).
    config : GAConfig
        Algorithm configuration.
    callbacks : list of callable, optional
        Functions ``f(gen, stats, population) -> bool``.

    Returns
    -------
    population : list
        Final population (carries the Pareto front).
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
    stats.register("max", np.max)

    for ind, fit in zip(population, map(toolbox.evaluate, population)):
        ind.fitness.values = fit

    # Assign initial crowding distance.
    population = toolbox.select(population, len(population))

    for gen in range(config.n_generations):
        t_gen_start = time.time()

        k_dcd = config.population_size
        if k_dcd % 4 != 0:
            k_dcd += 4 - (k_dcd % 4)
        pop_for_dcd = population[:]
        while len(pop_for_dcd) < k_dcd:
            pop_for_dcd.append(toolbox.clone(random.choice(population)))

        offspring = tools.selTournamentDCD(pop_for_dcd, k_dcd)[: config.population_size]
        offspring = [toolbox.clone(ind) for ind in offspring]

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

        population[:] = toolbox.select(population + offspring, config.population_size)

        fits_cv = [ind.fitness.values[0] for ind in population]
        n_feats = [_n_features_of(ind) for ind in population]
        best_ind = max(population, key=lambda ind: ind.fitness.values[0])

        gen_stats = EvolutionStats(
            generation=gen,
            best_fitness=float(best_ind.fitness.values[0]),
            mean_fitness=float(np.mean(fits_cv)),
            std_fitness=float(np.std(fits_cv)),
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

    return population, logbook, history
