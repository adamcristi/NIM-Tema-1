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
from operators.crossover.single_cut import single_cut_crossover
from operators.crossover.double_cut import double_cut_crossover


from path import LOGS_PATH, path, name

data, n_samples, n_candidates, total_used_sum = read_data(path)

data_matrix = preprocess_data(data, n_samples, n_candidates)

pop_size = 100

# Genetic Algorithm with Elitism

if sys.version_info.major == 3 and sys.version_info.minor >= 7:
    start = time.time_ns()
else:
    start = time.time()

last_population = ga_ep_elitism(pop_size=pop_size,
                                chromosome_size=n_candidates,
                                max_iterations=200,
                                mutation_prob=0.01,
                                mutation_choosing_prob=0.1,
                                crossover_prob=0.3,
                                pressure=6,
                                data_matrix=data_matrix,
                                eval_chromosome=eval_chromosome_ep_1,
                                crossover=single_cut_crossover,
                                percentage_elitism=0.05,  # 5% of population size are being kept
                                title=name[:-4],
                                logging=True)

if sys.version_info.major == 3 and sys.version_info.minor >= 7:
    end = time.time_ns()
    print(f"Total time: {(end-start) / 1e9} seconds.")
else:
    end = time.time()
    print(f"Total time: {(end - start)} seconds.")

