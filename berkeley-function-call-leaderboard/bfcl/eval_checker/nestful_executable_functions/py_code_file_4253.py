import random

random.seed(42)
from typing import Union



def random_boolean(p: Union[float, int]) -> bool:

    """Generates a random boolean value with a given probability `p`.

    The function returns True with probability `p` and False with probability `1 - p`.



    Args:

        p: The probability of returning True.

    """

    random_value = random.uniform(0, 1)

    return random_value < p

