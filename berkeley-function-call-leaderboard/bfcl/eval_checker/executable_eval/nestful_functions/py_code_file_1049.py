from typing import List



def vector_addition(vector1: List[float], vector2: List[float]) -> List[float]:

    """Performs vector addition between two vectors.



    Args:

        vector1: The first vector.

        vector2: The second vector.



    Raises:

        TypeError: If either argument is not a list.

        ValueError: If the two vectors are not of the same length.

    """

    if not isinstance(vector1, list) or not isinstance(vector2, list):

        raise TypeError("Both arguments must be of type list.")

    if len(vector1) != len(vector2):

        raise ValueError("The two vectors must be of the same length.")

    result = []

    for i in range(len(vector1)):

        result.append(vector1[i] + vector2[i])



    return result

