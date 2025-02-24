import math



def sqrt_function(x: float) -> float:

    """

    Implements the square root function using the formula:

    sqrt(x) = x^(1/2) = exp(ln(x) / 2)



    Args:

        x: The number to calculate the square root of.

    """

    return math.exp(math.log(x) / 2)

