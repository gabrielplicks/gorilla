import random

random.seed(42)
from typing import Union



def generate_bool() -> Union[bool]:

    """Generates a boolean value randomly, where the probability of True and False is equal.



    Args:

        None



    Returns:

        A boolean value.

    """

    return random.random() >= 0.5

