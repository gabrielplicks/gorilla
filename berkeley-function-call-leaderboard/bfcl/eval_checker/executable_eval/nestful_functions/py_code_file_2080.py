from typing import List, Tuple



def calc_fitness(chromosomes: List[Tuple[List, float]], max_fitness: float) -> List[float]:

    """Calculates the fitness of a given solution using the Crowding Distance method.

    Args:

        chromosomes: A list of chromosomes, where each chromosome is a tuple of individuals and corresponding fitness scores.

        max_fitness: The maximum fitness value among all chromosomes.

    """

    fitness_scores = []



    for chromosome in chromosomes:

        individuals, fitness_score = chromosome

        normalized_fitness = fitness_score / max_fitness

        fitness_scores.append(normalized_fitness)



    return fitness_scores

