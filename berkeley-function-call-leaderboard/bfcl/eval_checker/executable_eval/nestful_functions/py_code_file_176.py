from typing import List



def find_smallest_greater_number(array: List[int], target: int) -> int:

    """Finds the smallest number in an array that is greater than a given target number.



    Args:

        array: The input array of integers.

        target: The target number.



    Returns:

        The smallest number in the array that is greater than the target number, or -1 if no such number exists.

    """

    smallest_greater = -1

    for num in array:

        if num > target:

            smallest_greater = num

            break

    return smallest_greater

