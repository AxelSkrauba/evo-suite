"""DEAP toolbox construction and the two evolutionary loops (single / NSGA-II)."""

from __future__ import annotations

import copy
import random
import time

import numpy as np
from deap import base, tools

from evo_ens.core.config import EvoEnsConfig, EvolutionStats
from evo_ens.core.evaluator import EnsembleFitnessEvaluator
from evo_ens.utils.deap_utils import DeapTypes

_WEIGHT_CLIP = 3.0


def build_toolbox(
    deap_types: DeapTypes,
    n_candidates: int,
    evaluator: EnsembleFitnessEvaluator,
    config: EvoEnsConfig,
) -> base.Toolbox:
    """Configure a DEAP toolbox for a ``2 * n_candidates``-gene individual.

    Genes are initialised uniformly in ``[0, 1]``: the first half are
    activation thresholds, the second half are raw (pre-normalization)
    weights.
    """
    indpb = config.resolved_mutation_indpb(n_candidates)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register(
        "individual",
        tools.initRepeat,
        deap_types.individual_cls,
        toolbox.attr_float,
        n=2 * n_candidates,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("clone", copy.deepcopy)
    toolbox.register("evaluate", evaluator)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=config.mutation_sigma, indpb=indpb)

    if config.mode == "multiobjective":
        toolbox.register("select", tools.selNSGA2)
    else:
        toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)

    return toolbox


def _clip_weight_genes(individual: list[float], n_candidates: int) -> None:
    """Clip the weight half of an individual to ``[-3, 3]`` in place (stability)."""
    for i in range(n_candidates, 2 * n_candidates):
        individual[i] = float(np.clip(individual[i], -_WEIGHT_CLIP, _WEIGHT_CLIP))


def _record_stats(
    generation: int, population: list, n_candidates: int, elapsed: float
) -> EvolutionStats:
    fitness_values = [ind.fitness.values[0] for ind in population]
    n_models = [sum(1 for b in ind[:n_candidates] if b > 0.5) for ind in population]
    best_ind = max(population, key=lambda ind: ind.fitness.values[0])
    return EvolutionStats(
        generation=generation,
        best_fitness=max(fitness_values),
        mean_fitness=float(np.mean(fitness_values)),
        std_fitness=float(np.std(fitness_values)),
        best_n_models=sum(1 for b in best_ind[:n_candidates] if b > 0.5),
        mean_n_models=float(np.mean(n_models)),
        elapsed_time=elapsed,
    )


def run_single(
    toolbox: base.Toolbox, n_candidates: int, config: EvoEnsConfig
) -> tuple[list, list[EvolutionStats]]:
    """Run the single-objective GA with tournament selection and elitism."""
    population = toolbox.population(n=config.population_size)
    for ind, fit in zip(population, map(toolbox.evaluate, population)):
        ind.fitness.values = fit

    history: list[EvolutionStats] = []
    best_fitness_hist: list[float] = []
    no_improve = 0

    for gen in range(config.n_generations):
        t_gen = time.time()
        elites = [toolbox.clone(e) for e in tools.selBest(population, config.elite_size)]

        offspring = [toolbox.clone(ind) for ind in toolbox.select(population, len(population))]
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < config.crossover_prob:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values
        for mutant in offspring:
            if random.random() < config.mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        for ind in offspring:
            _clip_weight_genes(ind, n_candidates)

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        offspring.sort(key=lambda ind: ind.fitness.values[0])
        for i, elite in enumerate(elites):
            offspring[i] = elite
        population[:] = offspring

        elapsed = time.time() - t_gen
        gen_stats = _record_stats(gen, population, n_candidates, elapsed)
        history.append(gen_stats)

        if config.verbose and (gen % 10 == 0 or gen == config.n_generations - 1):
            print(repr(gen_stats))

        if config.early_stopping_rounds:
            best_fitness_hist.append(gen_stats.best_fitness)
            if len(best_fitness_hist) > config.early_stopping_rounds:
                window = best_fitness_hist[-config.early_stopping_rounds :]
                if max(window) - min(window) < config.early_stopping_tol:
                    no_improve += 1
                    if no_improve >= config.early_stopping_rounds:
                        if config.verbose:
                            print(f"  Early stopping at gen {gen}")
                        break
                else:
                    no_improve = 0

    return population, history


def run_nsga2(
    toolbox: base.Toolbox, n_candidates: int, config: EvoEnsConfig
) -> tuple[list, list[EvolutionStats]]:
    """Run the NSGA-II multiobjective GA over ``(score, compression)``."""
    population = toolbox.population(n=config.population_size)
    for ind, fit in zip(population, map(toolbox.evaluate, population)):
        ind.fitness.values = fit
    population = toolbox.select(population, len(population))  # assign crowding distance

    history: list[EvolutionStats] = []

    for gen in range(config.n_generations):
        t_gen = time.time()
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < config.crossover_prob:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values
        for mutant in offspring:
            if random.random() < config.mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        for ind in offspring:
            _clip_weight_genes(ind, n_candidates)

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        combined = population + offspring
        population[:] = toolbox.select(combined, config.population_size)

        elapsed = time.time() - t_gen
        gen_stats = _record_stats(gen, population, n_candidates, elapsed)
        history.append(gen_stats)

        if config.verbose and (gen % 10 == 0 or gen == config.n_generations - 1):
            print(repr(gen_stats))

    return population, history
