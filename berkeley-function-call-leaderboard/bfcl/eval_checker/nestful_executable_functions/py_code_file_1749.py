from math import sqrt

from typing import List



def calc_vector_distances(vector_a: List[float], vector_b: List[float]) -> List[float]:

    """Calculates the Euclidean distances between two vectors.



    Args:

        vector_a: A list representing a vector in n-dimensional space.

        vector_b: A list representing a vector in n-dimensional space.



    Returns:

        A list of distances, where each element in the list represents the distance between the corresponding elements in the two vectors.

    """

    return [sqrt((a - b) ** 2) for a, b in zip(vector_a, vector_b)]

