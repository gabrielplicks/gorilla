from typing import List



def multiples_of_3(numbers: List[int]) -> List[int]:

    """Returns a new list with only the numbers that are multiples of 3.



    Args:

        numbers: A list of numbers.



    Returns:

        A new list with only the numbers that are multiples of 3.

    """

    return list(filter(lambda x: x % 3 == 0, numbers))

