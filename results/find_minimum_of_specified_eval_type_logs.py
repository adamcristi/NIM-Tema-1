import os
import numpy as np

logs_folder = "ep"
#logs_folder = "rep"
#logs_folder = "elitism"
#logs_folder = "hybridized_hill_climbing"
#logs_folder = "hybridized_simulated_annealing"
#logs_folder = "hybridized_hill_climbing_elitism"
#logs_folder = "hybridized_simulated_annealing_elitism"

pop_size = 100

# LOGS_PATH is a must to be the absolute path to the logs
LOGS_PATH = os.path.join(os.path.split(os.path.abspath(os.getcwd()))[0], 'logs', logs_folder)


def filter_data(value):
    if value == '' or value == ';' or value == '=':
        return False
    else:
        return True


def find_minimum_of_specified_eval_type_logs(type_eval_chromosome):
    global_min_val = None
    generation_global_min_val = None
    log_name_global_min_val = None

    for root, dirs, files in os.walk(LOGS_PATH):
        for log_name in files:
            if 'parameters' not in log_name: # and (type_alg + '_1') in log_name:
                log_parameters_path = os.path.join(root, log_name[:-4] + "_parameters.txt")

                with open(log_parameters_path, "r") as fd:
                    log_parameters = fd.read(2048)

                if type_eval_chromosome in log_parameters:
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
                            current_covered = int(data_current_generation[data_current_generation.index('covered') + 1])
                            current_is_min_covered = int(data_current_generation[data_current_generation.index('is_min_covered') + 1])
                            current_min_val = int(data_current_generation[data_current_generation.index('min_val') + 1])

                            if current_is_min_covered == 1 and current_covered == pop_size:
                                if global_min_val is None:
                                    global_min_val = current_min_val
                                    generation_global_min_val = current_index_generation
                                    log_name_global_min_val = log_name
                                elif current_min_val < global_min_val:
                                    global_min_val = current_min_val
                                    generation_global_min_val = current_index_generation
                                    log_name_global_min_val = log_name

                            current_generation = file.readline().strip()

    return {'type_eval': type_eval_chromosome, 'global_min_val':global_min_val,
            'iteration_global_min_val':generation_global_min_val, 'log_name_global_min_val':log_name_global_min_val}


def find_parameters(log_name):
    with open(os.path.join(LOGS_PATH, log_name[:-4] + "_parameters.txt"), "r") as fd:
        log_parameters = fd.read(2048)

    return log_parameters


def find_min_data(log_name, index_min):
    with open(os.path.join(LOGS_PATH, log_name), "r") as fd:
        current_generation = fd.readline().strip()

        index_line = 0
        while current_generation:
            if index_line == index_min:
                data_current_generation = current_generation.split(" ")
                data_current_generation = list(filter(filter_data, data_current_generation))
                data_current_generation[0] = data_current_generation[0].replace(':', '')
                if "=" in data_current_generation[-1]:
                    data_current_generation[-1] = data_current_generation[-1].replace('=', '')

                min_data = data_current_generation

            index_line += 1
            current_generation = fd.readline().strip()

    return min_data


# Find smallest eval chromosome of population from selected logs
for val in [1, 2, 4]:
    data = find_minimum_of_specified_eval_type_logs(type_eval_chromosome='eval_chromosome_ep_{}'.format(str(val)))
    print(data)
    parameters = find_parameters(log_name=data["log_name_global_min_val"])
    min_data = find_min_data(log_name=data["log_name_global_min_val"], index_min=data['iteration_global_min_val'])
    print(parameters)
    print(min_data)
    print()


#data = find_minimum_of_specified_eval_type_logs(type_eval_chromosome='eval_chromosome_rep_1')
#print(data)
#parameters = find_parameters(log_name=data["log_name_global_min_val"])
#min_data = find_min_data(log_name=data["log_name_global_min_val"], index_min=data['iteration_global_min_val'])
#print(parameters)
#print(min_data)



#print(find_minimum_of_specified_eval_type_logs(type_eval_chromosome='eval_chromosome_ep_1'))
#print(find_minimum_of_specified_eval_type_logs(type_eval_chromosome='eval_chromosome_ep_2'))
#print(find_minimum_of_specified_eval_type_logs(type_eval_chromosome='eval_chromosome_ep_4'))

#print(find_minimum_of_specified_eval_type_logs(type_eval_chromosome='eval_chromosome_rep_1'))



#print(find_minimium_of_specified_alg_type_logs(type_alg='ga_ep'))
#print(find_minimium_of_specified_alg_type_logs(type_alg='ga_ep_elitism'))
#print(find_minimium_of_specified_alg_type_logs(type_alg='ga_ep_hybridized'))
##print(find_minimium_of_specified_alg_type_logs(type_alg='ga_ep_hybridized_elitism'))
#print(find_minimium_of_specified_alg_type_logs(type_alg='ga_ep_hybridized_simulated_annealing_elitism'))
#print(find_minimium_of_specified_alg_type_logs(type_alg='ga_ep_hybridized_hill_climbing_elitism'))