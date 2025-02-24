from typing import List



def mse(vector1: List[float], vector2: List[float]) -> float:

    """Calculates the mean squared error between two vectors of equal length.



    Args:

        vector1: The first vector.

        vector2: The second vector.



    Raises:

        AssertionError: If the vectors are not of equal length.

    """

    assert len(vector1) == len(vector2), "Vectors must be of equal length"

    differences = []

    for i in range(len(vector1)):

        differences.append((vector1[i] - vector2[i]) ** 2)

    return sum(differences) / len(differences)

