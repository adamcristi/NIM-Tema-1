import numpy as np
import pandas as pd

import sys
import os
from datetime import datetime
import time

from utils.preprocess_data import preprocess_data
from utils.read_data import read_data
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


# Genetic Algorithm with Elitism
def execute_ga_elitism(**kwargs):
    if "pop_size" not in kwargs.keys():
        kwargs["pop_size"] = 100

        population = np.random.randint(0, 2, (kwargs["pop_size"], n_candidates))

        if sys.version_info.major == 3 and sys.version_info.minor >= 7:
            start = time.time_ns()
        else:
            start = time.time()

        last_population = ga_ep_elitism(pop_size=kwargs["pop_size"],
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
                                        percentage_elitism=kwargs["percentage_elitism"],
                                        title=name[:-4],
                                        logging=True)

        if sys.version_info.major == 3 and sys.version_info.minor >= 7:
            end = time.time_ns()
            print(f"Total time: {(end-start) / 1e9} seconds.")
        else:
            end = time.time()
            print(f"Total time: {(end - start)} seconds.")

        print()


# SINGLE CUT CROSSOVER

# pressure - 1, 2, 4, 6 ;  eval_chromosome_ep_1

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_1,
                   crossover=single_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=1,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_1,
                   crossover=single_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=2,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_1,
                   crossover=single_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=4,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_1,
                   crossover=single_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=6,
                   percentage_elitism=0.05)  # 5% of population size are being kept

# pressure - 1, 2, 4, 6 ;  eval_chromosome_ep_2

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_2,
                   crossover=single_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=1,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_2,
                   crossover=single_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=2,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_2,
                   crossover=single_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=4,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_2,
                   crossover=single_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=6,
                   percentage_elitism=0.05)  # 5% of population size are being kept

# pressure - 1, 2, 4, 6 ;  eval_chromosome_ep_4

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_4,
                   crossover=single_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=1,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_4,
                   crossover=single_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=2,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_4,
                   crossover=single_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=4,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_4,
                   crossover=single_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=6,
                   percentage_elitism=0.05)  # 5% of population size are being kept

# DOUBLE CUT CROSSOVER

# pressure - 1, 2, 4, 6 ;  eval_chromosome_ep_1

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_1,
                   crossover=double_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=1,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_1,
                   crossover=double_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=2,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_1,
                   crossover=double_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=4,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_1,
                   crossover=double_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=6,
                   percentage_elitism=0.05)  # 5% of population size are being kept

# pressure - 1, 2, 4, 6 ;  eval_chromosome_ep_2

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_2,
                   crossover=double_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=1,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_2,
                   crossover=double_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=2,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_2,
                   crossover=double_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=4,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_2,
                   crossover=double_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=6,
                   percentage_elitism=0.05)  # 5% of population size are being kept

# pressure - 1, 2, 4, 6 ;  eval_chromosome_ep_4

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_4,
                   crossover=double_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=1,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_4,
                   crossover=double_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=2,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_4,
                   crossover=double_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=4,
                   percentage_elitism=0.05)  # 5% of population size are being kept

execute_ga_elitism(eval_chromosome=eval_chromosome_ep_4,
                   crossover=double_cut_crossover,
                   max_iterations=1000,
                   mutation_prob=0.01,
                   mutation_choosing_prob=0.1,
                   crossover_prob=0.3,
                   pressure=6,
                   percentage_elitism=0.05)  # 5% of population size are being kept