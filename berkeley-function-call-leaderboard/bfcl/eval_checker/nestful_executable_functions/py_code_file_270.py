from typing import List



def all_even(lst: List[int]) -> bool:

    """Checks if all elements in a list are even numbers.



    Args:

        lst: A list of integers.



    Returns:

        True if all elements in the list are even numbers, False otherwise.

    """

    for item in lst:

        if item % 2 != 0:

            return False

    return True

