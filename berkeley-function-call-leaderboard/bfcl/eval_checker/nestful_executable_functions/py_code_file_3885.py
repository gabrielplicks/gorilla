from typing import List



def compute_degree(coefficients: List[float]) -> int:

    """Computes the degree of a polynomial expressed as a list of coefficients in decreasing order of powers.

    The function handles a leading zero case where the degree of the polynomial is 0 (i.e., the list only contains a single zero).

    Args:

        coefficients: The coefficients of the polynomial in decreasing order of powers.

    """

    degree = len(coefficients) - 1

    if coefficients[0] == 0:

        degree = 0

    return degree

