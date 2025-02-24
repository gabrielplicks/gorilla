from typing import List



def check_list_for_value(numbers: List[int]) -> int:

    """Checks if a given list of numbers contains a value equal to or greater than 10.

    If a number is found, it returns that number, otherwise it returns 0.

    Args:

        numbers: A list of integers.

    """

    for num in numbers:

        if num >= 10:

            return num

    return 0

