import numpy as np
import pandas as pd

from datetime import datetime
import time

from ga_ep.preprocess_data import preprocess_data
from ga_ep.read_data import read_data
from ga_ep.ga_ep import ga_ep

from path import LOGS_PATH, path

data, n_samples, n_candidates, total_used_sum = read_data(path)

data_matrix = preprocess_data(data, n_samples, n_candidates)

pop_size = 100
population = np.random.randint(0, 2, (pop_size, n_candidates))

start = time.time_ns()
last_population = ga_ep(pop_size=pop_size,
                        chromosome_size=n_candidates,
                        max_iterations=100,
                        mutation_prob=0.01,
                        mutation_choosing_prob=0.1,
                        crossover_prob=0.3,
                        pressure=4,
                        data_matrix=data_matrix,
                        total_used_sum=total_used_sum,
                        population=population,
                        title="AC_01_cover",
                        logging=True)

end = time.time_ns()
print(f"Total time: {(end-start) / 1e9} seconds.")
