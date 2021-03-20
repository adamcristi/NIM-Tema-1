from utils.coverage_check import detailed_coverage_check
import numpy as np


def eval_chromosome_ep_1(chromosome, data_matrix):
    counts, full_coverage = detailed_coverage_check(chromosome, data_matrix)

    if not full_coverage:
        return np.count_nonzero(counts) / data_matrix.shape[0]

    else:
        ratio_used = np.count_nonzero(chromosome) / data_matrix.shape[1]
        return 2 - ratio_used
