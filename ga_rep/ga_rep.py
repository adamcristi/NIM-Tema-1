import numpy as np


def eval_chromosome_rep_1(chromosome, data_matrix):
    return 1 - (np.count_nonzero(chromosome) / data_matrix.shape[1])
