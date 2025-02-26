import random

random.seed(42)
from typing import Dict



def random_key_with_probability(dictionary: Dict[str, float]) -> str:

    """Randomly selects a key from the dictionary with a probability proportional to the value of the corresponding key.

    If all keys have negative values, the function returns a random key.



    Args:

        dictionary: A dictionary of string keys and numeric values.

    """

    total_sum = sum(dictionary.values())

    keys = []

    for key, value in dictionary.items():

        if value > 0:

            keys.extend([key] * value)

        else:

            keys.append(key)

    return random.choice(keys)

