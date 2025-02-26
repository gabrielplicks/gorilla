import random

random.seed(42)
from typing import Union



def coin_toss(p: Union[float, int]) -> str:

    """Simulates a coin toss and returns the outcome.

    Args:

        p: The probability of getting a head.

    """

    random_number = random.uniform(0, 1)

    if random_number < p:

        return 'Heads'

    else:

        return 'Tails'

