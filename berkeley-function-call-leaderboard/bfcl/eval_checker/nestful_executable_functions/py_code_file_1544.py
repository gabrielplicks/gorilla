from math import ceil

from random import sample

import random
random.seed(42)
from typing import List



def select_population_units(n: int, f: float) -> List[int]:

    """Selects a set of population units from a population of `n` units based on a given sampling fraction `f`.



    Args:

        n: The size of the population.

        f: The sampling fraction.



    Returns:

        A list of the selected population units.

    """

    n_selected = ceil(n * f)

    return sample(range(n), n_selected)

