import math



def cos_n(n: float) -> float:

    """Calculates the cosine of a numeric constant `n` using the `math` module.



    Args:

        n: The numeric constant to calculate the cosine of.



    Raises:

        TypeError: If the input is not a numeric constant.

    """

    if not isinstance(n, (int, float)):

        raise TypeError("The input must be a numeric constant")

    return math.cos(n)

