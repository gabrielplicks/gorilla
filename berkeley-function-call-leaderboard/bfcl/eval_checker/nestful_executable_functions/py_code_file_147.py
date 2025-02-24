import math

from typing import List, Tuple



def eucledian_distance(a: List[float], b: List[float]) -> float:

    """Calculates the Euclidean distance between two vectors.



    Args:

        a: The first vector.

        b: The second vector.



    Raises:

        Exception: If the vectors a and b are not of the same length.

    """

    if len(a) != len(b):

        raise Exception("Vectors a and b must be of the same length.")



    differences = [(a_i - b_i) ** 2 for a_i, b_i in zip(a, b)]

    sum_of_square_differences = sum(differences)

    euclidean_distance = math.sqrt(sum_of_square_differences)



    return euclidean_distance

