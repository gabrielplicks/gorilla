import math

from typing import List



def square_root_list(numbers: List[int]) -> List[float]:

    """Calculates the square root of each integer in a list.



    Args:

        numbers: A list of integers.



    Returns:

        A list of the square root of each integer in the input list.

    """

    return [math.sqrt(n) for n in numbers]

