from typing import List, Optional



def find_largest_less_than(numbers: List[int], value: int) -> Optional[int]:

    """Finds the largest integer in a list that is less than a given value.



    Args:

        numbers: A list of integers.

        value: The value to compare against.



    Returns:

        The largest integer in the list that is less than the given value, or None if no such integer exists.

    """

    max_less = None

    for num in numbers:

        if num < value:

            if max_less is None or num > max_less:

                max_less = num

    return max_less

