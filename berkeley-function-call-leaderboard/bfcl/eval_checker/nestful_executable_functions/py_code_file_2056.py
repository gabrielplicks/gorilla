from typing import List



def filter_out_negatives(input_list: List[int]) -> tuple:

    """Filters out negative numbers from the input list and returns a new list that contains only the positive numbers from the input list.

    The function also tracks the indices of the negative numbers in the input list and outputs a list of those indices.



    Args:

        input_list: A list of integers.



    Returns:

        A tuple containing a list of positive numbers and a list of indices of negative numbers.

    """

    indices = []

    positive_nums = [num for i, num in enumerate(input_list) if num >= 0]

    for i, num in enumerate(input_list):

        if num < 0:

            indices.append(i)

    return positive_nums, indices

