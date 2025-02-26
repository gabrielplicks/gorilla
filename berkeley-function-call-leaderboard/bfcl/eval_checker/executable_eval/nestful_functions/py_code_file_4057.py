import math

from typing import Union



def sum_power_series(x: Union[int, float], precision: float = 1e-9) -> float:

    """Calculates the sum of the power series ∑n=0∞x^n/n! until the partial sum reaches the specified precision.



    Args:

        x: A real number.

        precision: A floating-point number that specifies the precision of the partial sum.

    """

    partial_sum = 0

    n = 0

    while True:

        partial_sum += (x**n) / math.factorial(n)

        if abs(x**n) / math.factorial(n) < precision:

            break

        n += 1

    return partial_sum

