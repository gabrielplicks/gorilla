from typing import List



def polynomial(a: List[float], x: float) -> float:

    """

    Calculates the value of a polynomial at the given x value.



    Args:

        a: The coefficients of the polynomial.

        x: The value at which to evaluate the polynomial.

    """

    result = 0

    for i in range(len(a)):

        result += a[i] * (x ** i)

    return result

