from typing import List



def missing_number(numbers: List[int]) -> int:

    """Finds the missing number in a list of integers.



    Args:

        numbers: A list of integers with exactly one missing number.



    Returns:

        The missing number in the list.

    """

    minimum = min(numbers)

    maximum = max(numbers)

    for i in range(minimum, maximum + 1):

        if i not in numbers:

            return i

    return None

