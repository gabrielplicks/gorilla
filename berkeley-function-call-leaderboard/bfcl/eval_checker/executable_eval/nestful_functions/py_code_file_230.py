import copy

from typing import List



def divisible_by_three(matrix: List[List[int]]) -> List[int]:

    """

    Returns a list of all the integers that are divisible by 3 from a 2D list of integers.

    The function also flattens and copies the original 2D list.



    Args:

        matrix: A 2D list of integers.



    Returns:

        A list of integers that are divisible by 3.

    """

    flat_list = [item for sublist in matrix for item in sublist]

    copied_list = copy.deepcopy(flat_list)

    divisible_numbers = [num for num in copied_list if num % 3 == 0]



    return divisible_numbers

