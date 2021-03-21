import os
import numpy as np

# LOGS_PATH is a must to be the absolute path to the logs
from path import LOGS_PATH


def filter_data(value):
    if value == '' or value == ';' or value == '=':
        return False
    else:
        return True


def find_n_minimums_of_all_logs(n):
    global_n_mins = []

    for root, dirs, files in os.walk(LOGS_PATH):
        for log_name in files:
            if 'parameters' not in log_name:
                log_path = os.path.join(root, log_name)

                with open(log_path, "r") as file:

                    current_generation = file.readline().strip()

                    while current_generation:
                        data_current_generation = current_generation.split(" ")
                        data_current_generation = list(filter(filter_data, data_current_generation))
                        data_current_generation[0] = data_current_generation[0].replace(':', '')
                        if "=" in data_current_generation[-1]:
                            data_current_generation[-1] = data_current_generation[-1].replace('=', '')

                        current_index_generation = int(data_current_generation[0])
                        current_is_min_covered = int(data_current_generation[data_current_generation.index('is_min_covered') + 1])
                        current_min_val = int(data_current_generation[data_current_generation.index('min_val') + 1])

                        if current_is_min_covered == 1:
                            if len(global_n_mins) < n:
                                global_n_mins.append((current_min_val, current_index_generation, log_name))
                                global_n_mins = sorted(global_n_mins, key=lambda pair: pair[0])
                            elif len(global_n_mins) == n:
                                if current_min_val < global_n_mins[n - 1][0]:
                                    global_n_mins[n - 1] = (current_min_val, current_index_generation, log_name)
                                    global_n_mins = sorted(global_n_mins, key=lambda pair: pair[0])

                        current_generation = file.readline().strip()

    return global_n_mins


# Find first n smallest evals chromosome of population from entire logs
n_mins = find_n_minimums_of_all_logs(n=10)

for min in n_mins:
    print(min)