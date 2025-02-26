from typing import List



def mean_2d(array: List[List[float]]) -> float:

    """Calculates the mean value of a 2-dimensional array.



    Args:

        array: A 2-dimensional array of numbers.



    Returns:

        The mean value of the flattened array.

    """

    flattened = [item for sublist in array for item in sublist]

    total = sum(flattened)

    return total / len(flattened)

