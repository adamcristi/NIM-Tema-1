import numpy as np


def double_cut_crossover(population, crossover_prob):
    pop_size = population.shape[0]
    chromosome_size = population.shape[1]

    # Select chromosomes for crossover (save their indecies)
    indecies = np.nonzero(np.random.rand(pop_size) < crossover_prob)[0]
    # If there are an odd number of selected chromosomes, just ignore the last
    cross_count = len(indecies) - (len(indecies) % 2)

    for index in range(0, cross_count, 2):
        # get indecies of the two chromosomes in the population
        first, second = indecies[index], indecies[index + 1]
        # choose 2 cutting points at random (avoiding doing the same as mutation)

        cut1, cut2 = 1, 1
        # avoid doing nothing (equal cuts) or mutation (1 difference between cuts)
        while np.abs(cut1 - cut2) < 2:
            cuts = [np.random.randint(2, chromosome_size - 2), np.random.randint(2, chromosome_size - 2)]
            cut1, cut2 = min(cuts), max(cuts)

        # apply crossover at the given cutting points (interchange middle part)
        tmp = population[first, cut1:cut2].copy()
        population[first, cut1:cut2], population[second, cut1:cut2] = population[second, cut1:cut2], tmp
