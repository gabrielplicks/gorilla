import math



def sinh(x: float) -> float:

    """Calculates the hyperbolic sine of a number x.



    Args:

        x: The number for which we want to calculate the hyperbolic sine.

    """

    return (math.exp(x) - math.exp(-x)) / 2

