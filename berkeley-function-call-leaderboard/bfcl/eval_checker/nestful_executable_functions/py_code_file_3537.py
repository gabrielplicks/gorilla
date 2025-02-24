from typing import List



def count_enemies(sequence: List[int], target: int) -> int:

    """

    Returns the number of elements in `sequence` that are less than `target`.



    Args:

        sequence: A list of integers.

        target: An integer.

    """

    left, right = 0, len(sequence) - 1



    while left <= right:

        mid = (left + right) // 2



        if sequence[mid] < target:

            left = mid + 1

        else:

            right = mid - 1



    return left

