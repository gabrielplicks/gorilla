import random

random.seed(42)
from typing import List, Optional



def get_random_value(values: List[int], bound: int) -> Optional[int]:

    """

    Returns a randomly selected value from the given list that is less than or equal to the bound.

    If no such value is found, returns None.



    Args:

        values: A list of possible values.

        bound: An upper bound for the selected value.

    """

    filtered_values = [x for x in values if x <= bound]

    if len(filtered_values) == 0:

        return None

    return random.choice(filtered_values)

