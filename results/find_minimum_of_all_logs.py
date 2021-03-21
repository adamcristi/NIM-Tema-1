import os
import numpy as np

# LOGS_PATH is a must to be the absolute path to the logs
from path import LOGS_PATH


def filter_data(value):
    if value == '' or value == ';' or value == '=':
        return False
    else:
        return True


def find_minimum_of_all_logs():
    global_min_val = None
    generation_global_min_val = None
    log_name_global_min_val = None

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
                            if global_min_val is None:
                                global_min_val = current_min_val
                                generation_global_min_val = current_index_generation
                                log_name_global_min_val = log_name
                            elif current_min_val < global_min_val:
                                global_min_val = current_min_val
                                generation_global_min_val = current_index_generation
                                log_name_global_min_val = log_name

                        current_generation = file.readline().strip()

    return {'global_min_val':global_min_val, 'iteration_global_min_val':generation_global_min_val,
            'log_name_global_min_val':log_name_global_min_val}


# Find smallest eval chromosome of population from entire logs
print(find_minimum_of_all_logs())