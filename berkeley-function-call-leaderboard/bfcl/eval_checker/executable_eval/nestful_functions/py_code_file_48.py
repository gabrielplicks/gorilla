from typing import List



def get_indices_of_min_absolute_difference(array: List[int], target: int) -> List[int]:

    """

    Returns a list of indices where the absolute difference between the array's element and the target is the smallest.

    Args:

        array: A list of integers.

        target: The target value.

    """

    min_diff = float('inf')

    min_diff_indices = []



    for i, element in enumerate(array):

        diff = abs(element - target)

        if diff < min_diff:

            min_diff = diff

            min_diff_indices = [i]

        elif diff == min_diff:

            min_diff_indices.append(i)



    return min_diff_indices

