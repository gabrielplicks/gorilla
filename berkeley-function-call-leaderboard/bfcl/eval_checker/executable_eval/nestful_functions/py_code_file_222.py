from typing import List



def add_polynomials(p1: List[int], p2: List[int]) -> List[int]:

    """Adds two polynomials represented as lists of coefficients.

    Performs modular arithmetic on the coefficients with a modulus of 10^9 + 7.

    Args:

        p1: The first polynomial represented as a list of coefficients.

        p2: The second polynomial represented as a list of coefficients.

    """

    modulus = 10**9 + 7

    return [(p1_coef + p2_coef) % modulus for p1_coef, p2_coef in zip(p1, p2)]

