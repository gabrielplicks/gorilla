from typing import List, Any



def add_up(x: int, nums: List[Any]) -> int:

    """Calculates the sum of a given integer and a list of numbers of any type.



    Args:

        x: The integer to be added.

        nums: The list of numbers to be added.



    Returns:

        The sum of the integer and the numbers in the list.

    """

    return x + sum(nums)

