from typing import List



def calc_dot_product(a: List[float], b: List[float]) -> float:

    """Calculates the dot product of two vectors.



    Args:

        a: The first vector.

        b: The second vector.

    """

    result = 0

    for i in range(min(len(a), len(b))):

        result += a[i] * b[i]



    return result

