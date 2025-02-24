from typing import List



def remove_even(numbers: List[int]) -> List[int]:

    """Removes all the even numbers from a list of integers.

    If all the numbers in the list are even, returns an empty list.

    Args:

        numbers: A list of integers.

    """

    return list(filter(lambda x: x % 2 == 1, numbers))

