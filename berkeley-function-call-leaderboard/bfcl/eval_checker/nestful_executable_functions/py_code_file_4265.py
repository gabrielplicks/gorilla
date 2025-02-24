from typing import List



def find_smallest(numbers: List[int]) -> int:

    """Finds the smallest element in a list of integers.



    Args:

        numbers: A list of integers.



    Raises:

        TypeError: If the input is not a list or is empty.

    """

    if not isinstance(numbers, list):

        raise TypeError("The input is not a list or is empty")

    if len(numbers) == 0:

        return None

    return min(numbers)

