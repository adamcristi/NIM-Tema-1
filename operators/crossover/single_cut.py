import numpy as np


def single_cut_crossover(population, crossover_prob):
    pop_size = population.shape[0]
    chromosome_size = population.shape[1]

    # Select chromosomes for crossover (save their indecies)
    indecies = np.nonzero(np.random.rand(pop_size) < crossover_prob)[0]
    # If there are an odd number of selected chromosomes, just ignore the last
    cross_count = len(indecies) - (len(indecies) % 2)

    for index in range(0, cross_count, 2):
        # get indecies of the two chromosomes in the population
        first, second = indecies[index], indecies[index + 1]
        # choose a cutting point at random (avoiding doing the same as mutation)
        cut = np.random.randint(2, chromosome_size - 2)

        # apply crossover at the given cutting point (interchange first halves)
        tmp = population[first, :cut].copy()
        population[first, :cut], population[second, :cut] = population[second, :cut], tmp
