import math

from typing import List



def vecnorm(v: List[float]) -> float:

    """Calculates the Euclidean (L2) norm of a vector `v`.



    Args:

        v: A list of numbers representing the vector.



    Returns:

        The Euclidean (L2) norm of the vector `v`.

    """

    norm = 0

    for element in v:

        norm += element ** 2

    return math.sqrt(norm)

