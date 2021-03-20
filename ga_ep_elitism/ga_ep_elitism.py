import time
import sys
import copy
from datetime import datetime

import numpy as np

from utils.coverage_check import detailed_coverage_check
from ga_ep.ga_ep_algorithm import get_log_info_str, eval_population, fitness, selection_wheel_of_fortune, mutation, execute_threads, MAX_WORKERS

from path import LOGS_PATH


def elitism(population, current_population, evals_current_population, number_chromosomes_kept_by_elitism):
    sorted_chromosomes_by_evals = np.array([chromosome for _, chromosome in sorted(zip(evals_current_population, current_population),
                                                                                   key=lambda pair: pair[0])])

    chromosomes_kept_by_elitism = sorted_chromosomes_by_evals[:number_chromosomes_kept_by_elitism, :]

    for chromosome in chromosomes_kept_by_elitism:
        position_chromosome_to_change = np.random.randint(low=0, high=population.shape[0])

        population[position_chromosome_to_change] = chromosome


def ga_ep_elitism(pop_size, chromosome_size, max_iterations, mutation_prob, mutation_choosing_prob, crossover_prob,
                  pressure, data_matrix, eval_chromosome, crossover, population=None, percentage_elitism=0.05,
                  hybridization_type=None, title="", logging=True):
    """
      Genetic algorithm with evaluation penalty to satisfy the constraint.

      pop_size = the size of the population (only if population is None).
      chromosome_size = the size of a chromosome (only if population is None).
      max_iterations = maximum number of iterations.
      mutation_choosing_prob = probability of choosing a chromosome for mutation.
      mutation_prob = probability of mutation (used on each gene of a mutation-selected chromosome).
      crossover_prob = probability of choosing a chromosome for crossover.
      pressure = the selection pressure factor
      data_matrix = the preprocessed data matrix
      eval_chromosome = the evaluation function to use (it includes the penalty)
      crossover = crossover function
      population = the population to be used; if None, it will be randomly generated
      percentage_elitism = the percentage of how many best chromosomes from current population are kept at every generation.
    """
    if population is None:
        experiment_name = title + "_experiment_ga_ep_elitism_" + str(datetime.timestamp(datetime.now())) + ".txt"
        population = np.random.randint(0, 2, (pop_size, chromosome_size))
    else:
        experiment_name = title + "_experiment_ga_ep_hybridized_elitism_" + str(datetime.timestamp(datetime.now())) + ".txt"
        population = population.copy()

    number_chromoses_kept_by_elitism = int(percentage_elitism * pop_size)

    if logging:
        with open(LOGS_PATH + experiment_name[:-4] + "_parameters.txt", "w") as file:
            parameters = f"{experiment_name[:-4]}\n" \
                         + f"pop_size={population.shape[0]}\n" \
                         + f"chromosome_size={population.shape[1]}\n" \
                         + f"max_iterations={max_iterations}\n" \
                         + f"mutation_choosing_prob={mutation_choosing_prob}\n" \
                         + f"mutation_prob={mutation_prob}\n" \
                         + f"crossover_prob={crossover_prob}\n" \
                         + f"pressure={pressure}\n" \
                         + f"eval_chromosome={eval_chromosome.__name__}\n" \
                         + f"crossover={crossover.__name__}\n" \
                         + f"percentage_elitism={percentage_elitism}"
            if hybridization_type is not None:
                parameters += f"\nhybridization_type={hybridization_type}"
            file.write(parameters)

    if sys.version_info.major == 3 and sys.version_info.minor >= 7:
        start = time.time_ns()
    else:
        start = time.time()

    for iteration in range(max_iterations):
        # Save current population
        current_population = copy.deepcopy(population)

        # Evaluate the population
        evals = eval_population(population, data_matrix, eval_chromosome)

        # Compute FITNESS values
        fitness_values = fitness(evals, pressure=pressure)

        # SELECTION
        population = selection_wheel_of_fortune(population, fitness_values)

        # MUTATION
        mutation(population, mutation_prob, mutation_choosing_prob)

        # CROSSOVER
        crossover(population, crossover_prob)

        # ELITISM
        elitism(population, current_population, evals, number_chromoses_kept_by_elitism)

        if sys.version_info.major == 3 and sys.version_info.minor >= 7:
            end = time.time_ns()
            print(f"Iteration {iteration} - Elapsed time: {(end - start) / 1e9} seconds.")
        else:
            end = time.time()
            print(f"Iteration {iteration} - Elapsed time: {(end - start)} seconds.")

        if logging:
            with open(LOGS_PATH + experiment_name, "a+") as file:
                file.write(get_log_info_str(iteration, population, data_matrix) + "\n")

    return population