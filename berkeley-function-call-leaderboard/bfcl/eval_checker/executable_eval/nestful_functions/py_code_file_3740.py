import math



def special_formula(x: float, y: float) -> float:

    """Calculates the value of the special formula P(x, y) = arctan(x / y).



    Args:

        x: The numerator of the fraction.

        y: The denominator of the fraction.



    Returns:

        The value of P(x, y).

    """

    p = math.atan(x / y)

    return p

