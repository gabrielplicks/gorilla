from typing import List

from math import sqrt



def get_standard_deviation(array: List[int], mean: float = None) -> float:

    """Calculates the standard deviation of an array of integers.



    Args:

        array: The array of integers.

        mean: The mean of the array. If not provided, it is calculated.



    Returns:

        The standard deviation of the array.



    Raises:

        ValueError: If the array is empty.

    """

    if not array:

        raise ValueError("Cannot calculate the standard deviation of an empty array.")

    if mean is None:

        mean = sum(array) / len(array)

    std_dev = sqrt(sum((x - mean)**2 for x in array) / len(array))

    return std_dev

