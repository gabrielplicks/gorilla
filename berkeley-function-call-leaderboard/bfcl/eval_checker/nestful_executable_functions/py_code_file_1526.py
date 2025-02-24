from typing import List



def min_gap(lst: List[int]) -> int:

    """Finds the minimum absolute difference between any two distinct elements in a list of integers.



    Args:

        lst: A list of integers.



    Returns:

        The minimum absolute difference between any two distinct elements in the list, or -1 if the list has fewer than two elements.

    """

    if len(lst) < 2:

        return -1



    min_diff = abs(lst[1] - lst[0])



    for i in range(1, len(lst) - 1):

        for j in range(i + 1, len(lst)):

            diff = abs(lst[j] - lst[i])

            if diff < min_diff:

                min_diff = diff



    return min_diff

