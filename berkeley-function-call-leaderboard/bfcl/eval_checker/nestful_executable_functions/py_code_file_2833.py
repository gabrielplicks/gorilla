from typing import List



def put_zeros_ones(input_list: List[int]) -> List[int]:

    """

    Takes a list of integers and returns another list in the same shape as the input list,

    but with all zeroes at the beginning and all ones at the end.



    Args:

        input_list: A list of integers.



    Returns:

        A list of integers with all zeroes at the beginning and all ones at the end.

    """

    input_list.sort(key=lambda x: x == 0)

    input_list.reverse()

    return input_list

