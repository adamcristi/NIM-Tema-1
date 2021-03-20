from utils.coverage_check import detailed_coverage_check
import numpy as np


def eval_chromosome_rep_2(chromosome, data_matrix, total_used_sum):
    counts, full_coverage = detailed_coverage_check(chromosome, data_matrix)

    ratio = np.sum(counts) / total_used_sum

    # Without penalty
    return 1 - ratio
