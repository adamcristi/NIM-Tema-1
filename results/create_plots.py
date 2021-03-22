import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

logs_folder = "ep"
#logs_folder = "rep"
#logs_folder = "elitism"
#logs_folder = "hybridized_hill_climbing"
#logs_folder = "hybridized_simulated_annealing"
#logs_folder = "hybridized_hill_climbing_elitism"
#logs_folder = "hybridized_simulated_annealing_elitism"

log_name = 'AC_01_cover_experiment_ga_ep1616333482.801356.txt'

# LOGS_PATH is a must to be the absolute path to the logs
LOGS_PATH = os.path.join(os.path.split(os.path.abspath(os.getcwd()))[0], 'logs', logs_folder)
PLOTS_PATH = os.path.join(os.path.split(os.path.abspath(os.getcwd()))[0], 'plots', logs_folder)


def filter_data(value):
    if value == '' or value == ';' or value == '=':
        return False
    else:
        return True


def create_plot():
    min_vals = []
    are_mins_covered = []

    with open(os.path.join(LOGS_PATH, log_name[:-4] + "_parameters.txt"), "r") as fd:
        log_parameters = fd.read(2048)

    type_eval_chromosome = ''
    if 'eval_chromosome_ep_1' in log_parameters:
        type_eval_chromosome = "First Evaluation Function"
    elif 'eval_chromosome_ep_2' in log_parameters:
        type_eval_chromosome = "Second Evaluation Function"
    elif 'eval_chromosome_ep_4' in log_parameters:
        type_eval_chromosome = "Third Evaluation Function"
    elif 'eval_chromosome_rep_1' in log_parameters:
        type_eval_chromosome = "Reparation Evaluation Function"

    with open(os.path.join(LOGS_PATH, log_name), "r") as fd:
        current_generation = fd.readline().strip()

        while current_generation:
            data_current_generation = current_generation.split(" ")
            data_current_generation = list(filter(filter_data, data_current_generation))
            data_current_generation[0] = data_current_generation[0].replace(':', '')
            if "=" in data_current_generation[-1]:
                data_current_generation[-1] = data_current_generation[-1].replace('=', '')

            #current_index_generation = int(data_current_generation[0])
            #current_covered = int(data_current_generation[data_current_generation.index('covered') + 1])
            current_is_min_covered = int(data_current_generation[data_current_generation.index('is_min_covered') + 1])
            current_min_val = int(data_current_generation[data_current_generation.index('min_val') + 1])

            min_vals.append(current_min_val)
            if current_is_min_covered == 1:
                are_mins_covered.append('Yes')
            else:
                are_mins_covered.append('No')

            current_generation = fd.readline().strip()

    data_df = np.array([np.arange(len(min_vals))]).reshape(len(min_vals), 1)
    data_df = np.append(data_df, np.array([min_vals]).reshape(len(min_vals), 1), axis=1)
    data_df = np.append(data_df, np.array([are_mins_covered]).reshape(len(min_vals), 1), axis=1)

    df = pd.DataFrame(data=data_df, columns=["Iterations", "Best Chromosome Minimum Candidates", "All Samples Covered"])

    figure = plt.figure(figsize=(6.1, 6.1))
    sns.set_style("darkgrid")
    #sns.scatterplot(x=np.arange(len(min_vals))[::10], y=min_vals[::10], hue=df.loc[::10, "All Samples Covered"], s=150)
    sns.lineplot(x=np.arange(len(min_vals))[::5], y=min_vals[::5]) #, hue=df.loc[::5, "All Samples Covered"])
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Best Chromosome Minimum Candidates", fontsize=12)
    plt.tick_params(labelsize=11)
    plt.title(type_eval_chromosome, fontsize=14)
    plt.savefig(os.path.join(PLOTS_PATH, log_name + "_plot.png"))
    #plt.show()


# Create plot for min_val at every generation in specified log file
create_plot()
