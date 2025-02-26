from typing import List



class Individual:

    def __init__(self, fitness: float):

        self.fitness = fitness



def select_ibea(individuals: List[Individual], k: int) -> List[Individual]:

    """Selects the best `k` individuals based on their `fitness` attribute.

    Args:

        individuals: A list of individuals.

        k: The number of individuals to select.

    """

    individuals.sort(key=lambda x: x.fitness, reverse=True)

    return individuals[:k]

