import concurrent.futures
import time
import sys
from datetime import datetime

import numpy as np

from ga_ep.ga_ep_algorithm import execute_threads, fitness, selection_wheel_of_fortune, mutation

from utils.coverage_check import fast_coverage_check

from path import LOGS_PATH

MAX_WORKERS = 8


def get_log_info_str(iteration, population, data_matrix):
    number_candidates_chromosomes = [np.count_nonzero(chromosome) for chromosome in population]

    covered_count = np.count_nonzero([fast_coverage_check(chromosome, data_matrix) for chromosome in population])

    masked_number_candidates_chromosomes = np.ma.masked_array(number_candidates_chromosomes,
                                                              mask=number_candidates_chromosomes == 0)
    index_minimum_number_candidates_chromosome = np.where(number_candidates_chromosomes ==
                                                          np.min(masked_number_candidates_chromosomes))[0][0]
    is_min_covered = fast_coverage_check(population[index_minimum_number_candidates_chromosome], data_matrix)

    delim = " ;" + " " * 4

    info = f"{iteration}: "
    info += f"covered ={covered_count:4}{delim}"
    info += f"is_min_covered ={is_min_covered:5}{delim}"
    info += f"min_val ={np.min(masked_number_candidates_chromosomes):6}{delim}"
    info += f"max_val ={np.max(masked_number_candidates_chromosomes):6}{delim}"
    info += f"mean_val ={np.mean(masked_number_candidates_chromosomes):10.2f}{delim}"
    info += f"std_val ={np.std(masked_number_candidates_chromosomes):6.2f}"

    return info


def eval_population(population, data_matrix, eval_chromosome):
    return execute_threads(population, MAX_WORKERS, eval_chromosome, data_matrix)
    # return [eval_chromosome(chromosome, data_matrix, total_used_sum) for chromosome in population]


def ga_ep(pop_size, chromosome_size, max_iterations, mutation_prob, mutation_choosing_prob, crossover_prob, pressure,
          data_matrix, eval_chromosome, crossover, population=None, hybridization_type=None, title="", logging=True):
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
      population = the population to be used; if None, it will be randomly generated.
    """
    if population is None:
        experiment_name = title + "_experiment_ga_ep_" + str(datetime.timestamp(datetime.now())) + ".txt"
        population = np.random.randint(0, 2, (pop_size, chromosome_size))
    else:
        if hybridization_type is not None:
            experiment_name = title + f"_experiment_ga_ep_hybridized_{hybridization_type}_" + \
                              str(datetime.timestamp(datetime.now())) + ".txt"
        else:
            experiment_name = title + "_experiment_ga_ep_" + str(datetime.timestamp(datetime.now())) + ".txt"
        population = population.copy()

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
                         + f"crossover={crossover.__name__}"
            if hybridization_type is not None:
                parameters += f"\nhybridization_type={hybridization_type}"
            file.write(parameters)

    if sys.version_info.major == 3 and sys.version_info.minor >= 7:
        start = time.time_ns()
    else:
        start = time.time()

    for iteration in range(max_iterations):
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
