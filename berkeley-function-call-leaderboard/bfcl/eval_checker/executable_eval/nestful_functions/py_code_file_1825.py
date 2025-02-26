from math import sqrt

from typing import List



def square_roots(numbers: List[float]) -> List[int]:

    """Returns a list containing the square root of each element in the input list, rounded to the nearest integer.



    Args:

        numbers: A list of numbers.



    Returns:

        A list of integers.

    """

    return [round(sqrt(num)) for num in numbers]

