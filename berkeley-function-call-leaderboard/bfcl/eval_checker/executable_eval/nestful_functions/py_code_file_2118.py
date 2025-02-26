from typing import List



def increment_list(numbers: List[int]) -> List[int]:

    """Increments each number in a list by one.



    Args:

        numbers: A list of numbers.



    Returns:

        A list with each number incremented by one.

    """

    return [x + 1 for x in numbers]

