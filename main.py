import numpy as np
import pandas as pd

from datetime import datetime
import time

from ga_ep.ga_ep_1 import eval_chromosome_ep_1
from ga_ep.ga_ep_algorithm import ga_ep
from ga_rep.ga_rep import eval_chromosome_rep_1
from ga_rep.ga_rep_algorithm import ga_rep
from operators.crossover.double_cut import double_cut_crossover
from utils.coverage_check import detailed_coverage_check
from utils.preprocess_data import preprocess_data
from utils.read_data import read_data

from path import LOGS_PATH, path, name

data, n_samples, n_candidates, total_used_sum = read_data(path)

data_matrix = preprocess_data(data, n_samples, n_candidates)


def execute_ga(ga_function, **kwargs):
    if "pop_size" not in kwargs.keys():
        kwargs["pop_size"] = 100

    population = np.random.randint(0, 2, (kwargs["pop_size"], n_candidates))

    start = time.time_ns()
    last_population = ga_function(pop_size=kwargs["pop_size"],
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
                                  title=name[:-4],
                                  logging=True)

    end = time.time_ns()
    print(f"Total time: {(end - start) / 1e9} seconds.")


execute_ga(ga_function=ga_ep,
           eval_chromosome=eval_chromosome_ep_1,
           crossover=double_cut_crossover,
           max_iterations=200,
           mutation_prob=0.01,
           mutation_choosing_prob=0.1,
           crossover_prob=0.3,
           pressure=4)

execute_ga(ga_function=ga_rep,
           eval_chromosome=eval_chromosome_rep_1,
           crossover=double_cut_crossover,
           max_iterations=200,
           mutation_prob=0.01,
           mutation_choosing_prob=0.1,
           crossover_prob=0.3,
           pressure=4)
