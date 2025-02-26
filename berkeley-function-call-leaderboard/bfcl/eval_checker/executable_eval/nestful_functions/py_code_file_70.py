from typing import List



def compute_two_variable_diff_in_list(list_a: List[int], list_b: List[int]) -> int:

    """Computes the sum of the absolute differences between each pair of values in two lists.



    Args:

        list_a: A list of integers representing values at a certain time point.

        list_b: A list of integers representing values at a different time point.



    Returns:

        An integer representing the total difference.

    """

    if not list_a or not list_b or len(list_a) != len(list_b):

        return 0

    sum_of_diff = 0

    for a, b in zip(list_a, list_b):

        diff = abs(a - b)

        sum_of_diff += diff

    return sum_of_diff

