import random

random.seed(42)
import typing



def randint_range(low: typing.Optional[int] = 1, high: typing.Optional[int] = 100) -> int:

    """Generates a random integer within a given range.



    Args:

        low: The lower bound of the range. If not specified, the range starts at 1.

        high: The upper bound of the range. If not specified, the range ends at 100.

    """

    return random.randint(low, high)

