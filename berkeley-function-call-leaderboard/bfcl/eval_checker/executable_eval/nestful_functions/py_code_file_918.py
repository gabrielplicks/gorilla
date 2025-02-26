from typing import List



def vector_multiplication(vector_a: List[int], vector_b: List[int]) -> int:

    """Calculates the dot product of two equal-length vectors.



    Args:

        vector_a: The first vector.

        vector_b: The second vector.



    Returns:

        The dot product of the two vectors.

    """

    product = [a * b for a, b in zip(vector_a, vector_b)]

    return sum(product)

