from typing import List



def get_absolute_values(numbers: List[int]) -> tuple:

    """Returns a tuple containing two lists: the original list and a new list with the absolute values of the original list.



    Args:

        numbers: A list of integers.



    Returns:

        A tuple containing two lists: the original list and a new list with the absolute values of the original list.

    """

    return numbers, [abs(num) for num in numbers]

