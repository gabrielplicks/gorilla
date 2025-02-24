from typing import List



def sum_coeffs(p1: List[float], p2: List[float]) -> List[float]:

    """Calculates the sum of two polynomials represented as lists of coefficients.

    Args:

        p1: The coefficients of the first polynomial.

        p2: The coefficients of the second polynomial.

    """

    return [c1 + c2 for c1, c2 in zip(p1, p2)]

