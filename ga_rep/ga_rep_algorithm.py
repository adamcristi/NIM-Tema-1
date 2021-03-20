import concurrent.futures
import time
from datetime import datetime

import numpy as np

from utils.coverage_check import detailed_coverage_check, fast_coverage_check

from path import LOGS_PATH

MAX_WORKERS = 12


def get_log_info_str(iteration, population, data_matrix):
    selected_candidates = [np.count_nonzero(chromosome) for chromosome in population]

    covered_count = np.count_nonzero([fast_coverage_check(chromosome, data_matrix) for chromosome in population])

    min_index = np.where(selected_candidates == np.min(selected_candidates))[0][0]
    is_min_covered = fast_coverage_check(population[min_index], data_matrix)

    delim = " ;" + " " * 4

    info = f"{iteration}: "
    info += f"covered ={covered_count:4}{delim}"
    info += f"is_min_covered ={is_min_covered:5}{delim}"
    info += f"min_val ={np.min(selected_candidates):6}{delim}"
    info += f"max_val ={np.max(selected_candidates):6}{delim}"
    info += f"mean_val ={np.mean(selected_candidates):10.2f}{delim}"
    info += f"std_val ={np.std(selected_candidates):6.2f}"

    return info


def execute_threads(array, max_workers, function, *func_args):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as th_executor:

        arr_size = len(array)

        workers = 0
        tasks = []

        still_running = True
        element_index = 0

        while still_running:
            full_usage = True

            # If is not the end of population
            if element_index < arr_size:

                if workers < max_workers:
                    full_usage = False
                    workers += 1

                    element = array[element_index]
                    tasks.append(th_executor.submit(function, element, *func_args))

                    element_index += 1

            if full_usage:

                done, not_done = concurrent.futures.wait(tasks, return_when=concurrent.futures.FIRST_COMPLETED)

                # Safety mechanism
                if len(not_done) == 0 and len(done) == 0:
                    still_running = False

                else:
                    for future in done:
                        # Append the result to evals
                        results += [future.result()]

                        # Remove from active tasks
                        tasks.remove(future)

                        # Mark the worker as free
                        workers -= 1

    return results


def eval_population(population, data_matrix, eval_chromosome):
    return execute_threads(population, MAX_WORKERS, eval_chromosome, data_matrix)
    # return [eval_chromosome(chromosome, data_matrix, total_used_sum) for chromosome in population]


def single_fitness(eval, min_eval, eval_diff, pressure):
    return (1 + (eval - min_eval) / eval_diff) ** pressure


def fitness(evals, pressure=4):
    min_eval, max_eval = min(evals), max(evals)
    eval_diff = max_eval - min_eval

    # return execute_threads(evals, MAX_WORKERS, single_fitness, min_eval, eval_diff, pressure)
    return [(1 + (eval - min_eval) / eval_diff) ** pressure for eval in evals]


def selection_wheel_of_fortune(population, fitness_values):
    intervals = np.cumsum(fitness_values) / np.sum(fitness_values)

    pop_size = population.shape[0]
    selected = np.random.rand(pop_size)
    selected_indecies = []

    for value in selected:
        for index, interval in enumerate(intervals):
            if value < interval:
                selected_indecies += [index]
                break

    return population[selected_indecies]


def mutation(population, mutation_prob, mutation_choosing_prob):
    pop_size, chromosome_size = population.shape

    # Select chromosomes for mutation (save their indecies)
    indecies = np.nonzero(np.random.rand(pop_size) < mutation_choosing_prob)[0]

    for index in indecies:
        # Create a mask with selected genes for mutation
        mutation_mask = np.array(np.random.rand(chromosome_size) < mutation_prob, dtype=np.byte)

        # Apply mutation
        population[index] = (population[index] + mutation_mask) % 2


def repair_chromosome(chromosome, data_matrix):
    counts, full_coverage = detailed_coverage_check(chromosome, data_matrix)

    if not full_coverage:
        # Get the samples not covered
        zero_indecies = np.where(counts == 0)[0]

        repairing_candidates = []

        # Choose at random a candidate from each sample not covered
        for index in zero_indecies:
            ones_indecies = np.where(data_matrix[index] == 1)[0]
            random_index = np.random.randint(0, len(ones_indecies))
            repairing_candidates += [ones_indecies[random_index]]

        # Add the selected candidates to the chromosome
        repairing_candidates = np.unique(repairing_candidates)
        chromosome[:, repairing_candidates] = 1

    return chromosome


def repairing_procedure(population, data_matrix):
    repaired_population = execute_threads(population, MAX_WORKERS, repair_chromosome, data_matrix)
    return np.array(repaired_population)

    # for chr_index in range(population.shape[0]):
    #     counts, full_coverage = detailed_coverage_check(population[chr_index], data_matrix)
    #
    #     # Get the samples not covered
    #     zero_indecies = np.where(counts == 0)[0]
    #
    #     repairing_candidates = []
    #
    #     # Choose at random a candidate from each sample not covered
    #     for index in zero_indecies:
    #         ones_indecies = np.where(data_matrix[index] == 1)[0]
    #         random_index = np.random.randint(0, len(ones_indecies))
    #         repairing_candidates += [ones_indecies[random_index]]
    #
    #     # Add the selected candidates to the chromosome
    #     repairing_candidates = np.unique(repairing_candidates)
    #     population[chr_index, repairing_candidates] = 1


def ga_rep(pop_size, chromosome_size, max_iterations, mutation_prob, mutation_choosing_prob, crossover_prob, pressure,
           data_matrix, eval_chromosome, crossover, population=None, title="", logging=True):
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
      eval_chromosome = the evaluation function to use for one chromosome
      crossover = crossover function
      population = the population to be used; if None, it will be randomly generated.
    """
    if population is None:
        population = np.random.randint(0, 2, (pop_size, chromosome_size))
    else:
        population = population.copy()

    experiment_name = title + "_experiment_ga_ep" + str(datetime.timestamp(datetime.now())) + ".txt"

    if logging:
        with open(LOGS_PATH + experiment_name[:-4] + "_parameters.txt", "w") as file:
            parameters = f"{experiment_name[:-4]}\n" \
                         + f"pop_size={population.shape[0]}\n" \
                         + f"chromosome_size={population.shape[1]}\n" \
                         + f"max_iterations={max_iterations}\n" \
                         + f"mutation_choosing_prob={mutation_choosing_prob}\n" \
                         + f"mutation_prob={mutation_prob}\n" \
                         + f"crossover_prob={crossover_prob}\n" \
                         + f"pressure={pressure}"
            file.write(parameters)

    start = time.time_ns()

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

        # REPAIR PROCEDURE
        population = repairing_procedure(population, data_matrix)

        end = time.time_ns()
        print(f"Iteration {iteration} - Elapsed time: {(end - start) / 1e9} seconds.")

        if logging:
            with open(LOGS_PATH + experiment_name, "a+") as file:
                file.write(get_log_info_str(iteration, population, data_matrix) + "\n")

    return population
