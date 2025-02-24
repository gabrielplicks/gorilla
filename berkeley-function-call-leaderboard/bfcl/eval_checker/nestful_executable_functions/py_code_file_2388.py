from typing import List



def product_of_vectors(vector1: List[float], vector2: List[float]) -> float:

    """Computes the product of two vectors.



    Args:

        vector1: The first vector.

        vector2: The second vector.



    Returns:

        The product of the two vectors.

    """

    assert len(vector1) == len(vector2), "Vector lengths must be equal"

    product = 0

    for i in range(len(vector1)):

        product += vector1[i] * vector2[i]

    return product

