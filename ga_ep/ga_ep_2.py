from utils.coverage_check import fast_coverage_check
import numpy as np


def eval_chromosome_ep_2(chromosome, data_matrix):
    full_coverage = fast_coverage_check(chromosome, data_matrix)

    ratio = np.count_nonzero(chromosome) / data_matrix.shape[1]

    return 1 - ratio if full_coverage else ratio
