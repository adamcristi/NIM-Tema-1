import concurrent.futures
import time
from datetime import datetime

import numpy as np

from ga_ep.coverage_check import detailed_coverage_check, fast_coverage_check

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


def eval_chromosome(chromosome, data_matrix, total_used_sum):
    counts, full_coverage = detailed_coverage_check(chromosome, data_matrix)

    ratio = np.sum(counts) / total_used_sum

    return 1 - ratio if full_coverage else ratio


def eval_population(population, data_matrix, total_used_sum):
    return execute_threads(population, MAX_WORKERS, eval_chromosome, data_matrix, total_used_sum)
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


def crossover(population, crossover_prob):
    pop_size = population.shape[0]
    chromosome_size = population.shape[1]

    # Select chromosomes for crossover (save their indecies)
    indecies = np.nonzero(np.random.rand(pop_size) < crossover_prob)[0]
    # If there are an odd number of selected chromosomes, just ignore the last
    cross_count = len(indecies) - (len(indecies) % 2)

    for index in range(0, cross_count, 2):
        # get indecies of the two chromosomes in the population
        first, second = indecies[index], indecies[index + 1]
        # choose a cutting point at random (avoiding doing the same as mutation)
        cut = np.random.randint(2, chromosome_size - 2)

        # apply crossover at the given cutting point (interchange first halves)
        tmp = population[first, :cut].copy()
        population[first, :cut], population[second, :cut] = population[second, :cut], tmp


def ga_ep(pop_size, chromosome_size, max_iterations, mutation_prob, mutation_choosing_prob, crossover_prob, pressure,
          data_matrix, total_used_sum, population=None, title="", logging=True):
    '''
      Genetic algorithm with evaluation penalty to satisfy the constraint.

      pop_size = the size of the population (only if population is None).
      chromosome_size = the size of a chromosome (only if population is None).
      max_iterations = maximum number of iterations.
      mutation_choosing_prob = probability of choosing a chromosome for mutation.
      mutation_prob = probability of mutation (used on each gene of a mutation-selected chromosome).
      crossover_prob = probability of choosing a chromosome for crossover.
      pressure = the selection pressure factor
      data_matrix = the preprocessed data matrix
      total_used_sum = the sum of used candidates per sample
      population = the population to be used; if None, it will be randomly generated.
    '''
    if population is None:
        population = np.random.randint(0, 2, (pop_size, chromosome_size))
    else:
        population = population.copy()

    experiment_name = title + "_experiment_" + str(datetime.timestamp(datetime.now())) + ".txt"

    start = time.time_ns()
    for iteration in range(max_iterations):
        # Evaluate the population
        evals = eval_population(population, data_matrix, total_used_sum)

        # Compute FITNESS values
        fitness_values = fitness(evals, pressure=pressure)

        # SELECTION
        population = selection_wheel_of_fortune(population, fitness_values)

        # MUTATION
        mutation(population, mutation_prob, mutation_choosing_prob)

        # CROSSOVER
        crossover(population, crossover_prob)

        end = time.time_ns()
        print(f"Iteration {iteration} - Elapsed time: {(end - start) / 1e9} seconds.")

        if logging:
            with open(LOGS_PATH + experiment_name, "a+") as file:
                file.write(get_log_info_str(iteration, population, data_matrix) + "\n")

    return population
