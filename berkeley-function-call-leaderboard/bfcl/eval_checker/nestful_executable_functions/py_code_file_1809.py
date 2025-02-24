import random

random.seed(42)
from typing import Dict



def weighted_random_key(weights: Dict[str, float]) -> str:

    """

    Returns a randomly selected key from the given dictionary based on the provided weights.



    Args:

        weights: A dictionary where keys are the options and values are the weights.

    """

    return random.choices(list(weights.keys()), weights=list(weights.values()))[0]

