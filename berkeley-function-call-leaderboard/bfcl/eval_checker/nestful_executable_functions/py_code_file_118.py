from typing import List



def P(x: float, a: List[float]) -> float:

    """Calculates the value of a polynomial at a given point `x`.



    Args:

        x: The point at which to evaluate the polynomial.

        a: The coefficients of the polynomial.

    """

    n = len(a) - 1  # degree of the polynomial

    p = 0

    for i in range(n+1):

        p += a[i] * x**i

    return p

