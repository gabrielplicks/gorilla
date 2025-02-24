from typing import List



def schur_product(a: List[float], b: List[float]) -> List[float]:

    """Calculates the Schur product (or Hadamard product) of two vectors.

    The Schur product for two vectors `a` and `b` is a vector `c` where each element is the product of the corresponding elements in `a` and `b`.

    Args:

        a: The first vector.

        b: The second vector.

    """

    return [ai * bi for ai, bi in zip(a, b)]

