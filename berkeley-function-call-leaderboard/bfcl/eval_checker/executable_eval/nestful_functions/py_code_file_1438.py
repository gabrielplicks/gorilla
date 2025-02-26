import random

random.seed(42)
from typing import Tuple



def randomize_number(seed: int, size: int = 10) -> Tuple[str, str]:

    """Generates a random number with a given seed and returns a tuple `(r, sr)`.



    Args:

        seed: An integer used for generating the random numbers.

        size: The desired length of the random number. Defaults to 10.



    Returns:

        A tuple containing the resulting randomized number as `r` and its slice of the first half as `sr`.

    """

    random.seed(seed)

    r = ''.join(random.choices('0123456789', k=size))

    r = ''.join(random.sample(r, k=size))

    return r, r[:size//2]

