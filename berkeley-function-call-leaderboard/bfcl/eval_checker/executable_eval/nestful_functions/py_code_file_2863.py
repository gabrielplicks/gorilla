from typing import List



def max_two_of_each(numbers: List[int]) -> List[int]:

    """Returns a list with a maximum of two of each distinct number.



    Args:

        numbers: A list of integers.



    Returns:

        A list with a maximum of two of each distinct number.

    """

    return [x for x in set(numbers) for _ in range(2)]

