import random

random.seed(42)
from typing import List



def crossover_uniform(parent1: List[int], parent2: List[int]) -> List[int]:

    """Performs uniform crossover on the given two lists of chromosomes.

    Returns a single child chromosome (as a list) created by randomly selecting elements from each parent at each position in the chromosome.

    Args:

        parent1: The first parent chromosome.

        parent2: The second parent chromosome.

    """

    child = []

    for i in range(len(parent1)):

        if random.random() < 0.5:

            child.append(parent1[i])

        else:

            child.append(parent2[i])

    return child

