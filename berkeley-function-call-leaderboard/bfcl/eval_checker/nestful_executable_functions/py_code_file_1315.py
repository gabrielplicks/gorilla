from typing import List



def any_even(numbers: List[int]) -> bool:

    """Checks if any of the numbers in a given list is even.



    Args:

        numbers: A list of integers.



    Returns:

        True if any of the numbers in the list are even, False otherwise.

    """

    return any([num % 2 == 0 for num in numbers])

