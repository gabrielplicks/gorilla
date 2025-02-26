from typing import Tuple



def find_min_max_avg(a: float, b: float, c: float) -> Tuple[float, float, float]:

    """Calculates the minimum, maximum, and average of three numbers.

    Args:

        a: The first number.

        b: The second number.

        c: The third number.

    """

    min_val = min(a, b, c)

    max_val = max(a, b, c)

    avg_val = sum([a, b, c]) / len([a, b, c])

    return (min_val, max_val, avg_val)

