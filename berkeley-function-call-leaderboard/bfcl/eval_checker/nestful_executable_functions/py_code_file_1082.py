from typing import List



def min_values(input_list: List[List[int]]) -> List[int]:

    """

    Returns a new list consisting of the minimum value at each index among all the input lists.



    Args:

        input_list: A list of lists of integers.

    """

    return [min(values) for values in zip(*input_list)]

