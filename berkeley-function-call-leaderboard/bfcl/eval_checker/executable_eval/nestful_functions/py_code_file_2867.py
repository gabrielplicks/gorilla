from typing import List

import random

random.seed(42)


def weighted_die(probabilities: List[float]) -> int:

    """Generates an integer in the range 1 to 6 that follows a weighted distribution.



    The input to the function is a list of non-negative numbers that sum to one,

    representing the probability of each outcome.



    Args:

        probabilities: A list of non-negative numbers that sum to one.



    Returns:

        An integer in the range 1 to 6, weighted by the given probabilities.



    Raises:

        ValueError: If the input probabilities do not sum to 1.

    """

    if sum(probabilities) != 1:

        raise ValueError('Error: The probabilities do not sum to 1.')

    return random.choices(range(1, 7), weights=probabilities)[0]

