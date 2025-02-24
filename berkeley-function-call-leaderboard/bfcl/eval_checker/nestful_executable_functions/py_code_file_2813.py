from typing import List



def get_positive_integer_index(input_list: List[int], n: int) -> int:

    """

    Returns the index of the first integer in the input list that is a multiple of n.

    If no such integer exists, returns -1.



    Args:

        input_list: A list of integers.

        n: A positive integer.



    Raises:

        ValueError: If the input list is empty or n is not a positive integer.

    """

    assert len(input_list) > 0, "The input list cannot be empty"

    assert n > 0, "n must be a positive integer"



    for i, val in enumerate(input_list):

        if val % n == 0:

            return i

    return -1

