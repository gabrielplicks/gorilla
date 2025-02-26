import random

random.seed(42)
from typing import List



def sample_items(items: List, num_items: int, replacement: bool = False) -> List:

    """Randomly samples a specified number of items from a list, with or without replacement.



    Args:

        items: The list of items to sample from.

        num_items: The number of items to sample.

        replacement: Whether to perform sampling with replacement. Defaults to False.



    Returns:

        A list of sampled items.

    """

    if replacement:

        return random.choices(items, k=num_items)

    else:

        return random.sample(items, k=num_items)

