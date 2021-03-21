import numpy as np
import pandas as pd

import sys
import os
from datetime import datetime
import time

from utils.preprocess_data import preprocess_data
from utils.read_data import read_data
from hill_climbing.hill_climbing import steepest_ascent_hill_climbing, next_ascent_hill_climbing
from simulated_annealing.simulated_annealing import simulated_annealing
from ga_ep_elitism.ga_ep_elitism import ga_ep_elitism
from ga_ep.ga_ep_1 import eval_chromosome_ep_1
from ga_ep.ga_ep_2 import eval_chromosome_ep_2
from ga_ep.ga_ep_3 import eval_chromosome_ep_3
from ga_ep.ga_ep_4 import eval_chromosome_ep_4
from operators.crossover.single_cut import single_cut_crossover
from operators.crossover.double_cut import double_cut_crossover

from path import LOGS_PATH, path, name

data, n_samples, n_candidates, total_used_sum = read_data(path)

data_matrix = preprocess_data(data, n_samples, n_candidates)

pop_size = 100


# Genetic Algorithm Hybridized with Elitism -> Hill Climbing / Simulated Annealing used to get a better initial
# population than a random one
def execute_ga_hybridized_elitism(**kwargs):
    if "pop_size" not in kwargs.keys():
        kwargs["pop_size"] = 100

    if sys.version_info.major == 3 and sys.version_info.minor >= 7:
        start = time.time_ns()
    else:
        start = time.time()

    # Hill Climbing - selecting best improving neighbour
    #population = next_ascent_hill_climbing(number_iterations=1, number_points=pop_size, data_matrix=data_matrix,
    #                                       number_samples=n_samples, number_candidates=n_candidates, total_used_sum=total_used_sum)

    # Hill Climbing - selecting first improving neighbour (takes way more time than best improving)
    #population = steepest_ascent_hill_climbing(number_iterations=1, number_points=pop_size, data_matrix=data_matrix,
    #                                           number_samples=n_samples, number_candidates=n_candidates, total_used_sum=total_used_sum)

    # Simulated Annealing
    #population = simulated_annealing(number_iterations=1, number_points=pop_size, data_matrix=data_matrix,number_samples=n_samples,
    #                                 number_candidates=n_candidates, total_used_sum=total_used_sum)


    # Load data Hill Climber because the algorithm takes too much time (for AC_01_cover ~ 1 hour and way more for the others)
    population = np.load(os.path.abspath(os.getcwd()) + '/hill_climbing/data_hill_climber/{}_{}_Instance_Best_Size_{}.npy'.
                         format(name.split('_')[0],
                                name.split('_')[1],
                                str(pop_size)))
    hybridization_type = "hill_climbing"

    # Load data Simulated Annealer because the algorithm takes too much time (for AC_01_cover ~ 1 hour and way more for the others)
    #population = np.load(os.path.abspath(os.getcwd()) + '/simulated_annealing/data_simulated_annealer/{}_{}_Instance_Size_{}.npy'.
    #                     format(name.split('_')[0],
    #                            name.split('_')[1],
    #                            str(pop_size)))
    #hybridization_type = "simulated_annealing"

    last_population = ga_ep_elitism(pop_size=pop_size,
                                    chromosome_size=n_candidates,
                                    max_iterations=kwargs["max_iterations"],
                                    mutation_prob=kwargs["mutation_prob"],
                                    mutation_choosing_prob=kwargs["mutation_choosing_prob"],
                                    crossover_prob=kwargs["crossover_prob"],
                                    pressure=kwargs["pressure"],
                                    data_matrix=data_matrix,
                                    eval_chromosome=kwargs["eval_chromosome"],
                                    crossover=kwargs["crossover"],
                                    population=population,
                                    percentage_elitism=kwargs["percentage_elitism="],
                                    hybridization_type=hybridization_type,
                                    title=name[:-4],
                                    logging=True)

    if sys.version_info.major == 3 and sys.version_info.minor >= 7:
        end = time.time_ns()
        print(f"Total time: {(end-start) / 1e9} seconds.")
    else:
        end = time.time()
        print(f"Total time: {(end - start)} seconds.")

    print()

# DOUBLE CUT CROSSOVER

# pressure - 6, 8 ;  eval_chromosome_ep_1

execute_ga_hybridized_elitism(eval_chromosome=eval_chromosome_ep_1,
                              crossover=double_cut_crossover,
                              max_iterations=1000,
                              mutation_prob=0.01,
                              mutation_choosing_prob=0.1,
                              crossover_prob=0.3,
                              pressure=6,
                              percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_hybridized_elitism(eval_chromosome=eval_chromosome_ep_1,
                              crossover=double_cut_crossover,
                              max_iterations=1000,
                              mutation_prob=0.01,
                              mutation_choosing_prob=0.1,
                              crossover_prob=0.3,
                              pressure=8,
                              percentage_elitism=0.05)  # 5% of population size are being kept

# pressure - 6, 8 ;  eval_chromosome_ep_2

execute_ga_hybridized_elitism(eval_chromosome=eval_chromosome_ep_2,
                              crossover=double_cut_crossover,
                              max_iterations=1000,
                              mutation_prob=0.01,
                              mutation_choosing_prob=0.1,
                              crossover_prob=0.3,
                              pressure=6,
                              percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_hybridized_elitism(eval_chromosome=eval_chromosome_ep_2,
                              crossover=double_cut_crossover,
                              max_iterations=1000,
                              mutation_prob=0.01,
                              mutation_choosing_prob=0.1,
                              crossover_prob=0.3,
                              pressure=8,
                              percentage_elitism=0.05)  # 5% of population size are being kept

# pressure - 6, 8 ;  eval_chromosome_ep_4

execute_ga_hybridized_elitism(eval_chromosome=eval_chromosome_ep_4,
                              crossover=double_cut_crossover,
                              max_iterations=1000,
                              mutation_prob=0.01,
                              mutation_choosing_prob=0.1,
                              crossover_prob=0.3,
                              pressure=6,
                              percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_hybridized_elitism(eval_chromosome=eval_chromosome_ep_4,
                              crossover=double_cut_crossover,
                              max_iterations=1000,
                              mutation_prob=0.01,
                              mutation_choosing_prob=0.1,
                              crossover_prob=0.3,
                              pressure=8,
                              percentage_elitism=0.05)  # 5% of population size are being kept
