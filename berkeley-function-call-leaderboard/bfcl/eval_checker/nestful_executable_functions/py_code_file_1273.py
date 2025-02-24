from typing import List



def get_three_slices(input_list: List[int]) -> List[List[int]]:

    """Returns a list of three slices from the input list, where each slice is of length 3.

    If the input list's length is less than 3, the function returns an empty list.

    Args:

        input_list: A list of integers.

    """

    slices = []

    if len(input_list) < 3:

        return []

    for i in range(len(input_list) - 2):

        slice = input_list[i:i+3]

        slices.append(slice)

    return slices

