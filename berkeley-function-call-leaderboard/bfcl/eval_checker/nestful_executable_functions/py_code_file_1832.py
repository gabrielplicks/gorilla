import random

random.seed(42)
import math



def generate_random_float() -> float:

    """Generates a random float between 0 and 1, inclusive.



    Args:

        None



    Returns:

        float: A random float between 0 and 1, inclusive.

    """

    x = random.uniform(0, 1)

    x = math.ceil(x) if x > 0.5 else math.floor(x)

    return x * math.log(2, 10)

