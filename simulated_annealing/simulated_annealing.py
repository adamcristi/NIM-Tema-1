import numpy as np
import copy
import os

from utils.coverage_check import detailed_coverage_check
from ga_ep.ga_ep_algorithm import execute_threads, MAX_WORKERS
from path import name


def eval_point(point, data_matrix, number_samples, total_used_sum):
    counts, full_coverage = detailed_coverage_check(point, data_matrix)

    if not full_coverage:
        ratio_no_coverage = 1 - (np.count_nonzero(counts) / number_samples)
        return 1 + ratio_no_coverage
    else:
        ratio_used = np.count_nonzero(point) / data_matrix.shape[1]
        return ratio_used


def eval_points(points, data_matrix, number_samples, total_used_sum):
    return execute_threads(points, MAX_WORKERS, eval_point, data_matrix, number_samples, total_used_sum)
    #return [eval_point(point, data_matrix, number_samples) for point in points]


def get_neighbours_point(point):
    neighbours_point = []

    for index, value in enumerate(point):
        neighbour = copy.deepcopy(point)
        neighbour[index] = 1 - value

        neighbours_point.append(neighbour)

    return neighbours_point


def simulated_annealing(number_iterations, number_points, data_matrix, number_samples, number_candidates, total_used_sum,
                        temperature=0.001, alpha=0.9):
    best_points = np.zeros((number_points, number_candidates), dtype=np.int64)
    evals_best_points = np.zeros((number_candidates,))

    for iteration in range(number_iterations):
        current_points = np.random.randint(0, 2, (number_points, number_candidates))

        for index_current_point, current_point in enumerate(current_points):
            updated_current_point = copy.deepcopy(current_point)
            evals_best_points[index_current_point] = eval_point(point=current_point, data_matrix=data_matrix,
                                                                number_samples=number_samples,
                                                                total_used_sum=total_used_sum)

            modified_minimum = True

            while modified_minimum:
                eval_current_point = eval_point(point=current_point, data_matrix=data_matrix,
                                                number_samples=number_samples,
                                                total_used_sum=total_used_sum)
                eval_updated_current_point = eval_current_point

                modified_minimum = False

                neighbours_current_point = get_neighbours_point(point=current_point)

                evals_neighbours_current_point = eval_points(points=neighbours_current_point, data_matrix=data_matrix,
                                                             number_samples=number_samples, total_used_sum=total_used_sum)

                for index_neighbour, eval_neighbour in enumerate(evals_neighbours_current_point):
                    if eval_neighbour < eval_current_point:
                        eval_updated_current_point = eval_neighbour
                        updated_current_point = neighbours_current_point[index_neighbour]
                    else:
                        random_value = np.random.rand()
                        if random_value < np.exp((eval_current_point - eval_neighbour) / temperature):
                            eval_updated_current_point = eval_neighbour
                            updated_current_point = neighbours_current_point[index_neighbour]

                if eval_updated_current_point < evals_best_points[index_current_point]:
                    evals_best_points[index_current_point] = eval_updated_current_point
                    best_points[index_current_point] = updated_current_point
                    modified_minimum = True

                temperature *= alpha

                current_point = copy.deepcopy(updated_current_point)

    np.save(os.path.abspath(os.getcwd()) + '/simulated_annealing/data_simulated_annealer/{}_{}_Instance_Size_{}'.format(name.split('_')[0],
                                                                                                                        name.split('_')[1],
                                                                                                                        str(number_points)), best_points)

    return best_points
