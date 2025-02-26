from typing import Union



def probability_at_least_one(p1: Union[float, int], p2: Union[float, int]) -> float:

    """Calculates the probability of at least one of two independent events occurring.



    Args:

        p1: The probability of the first event occurring.

        p2: The probability of the second event occurring.

    """

    return p1 + p2 - p1 * p2

