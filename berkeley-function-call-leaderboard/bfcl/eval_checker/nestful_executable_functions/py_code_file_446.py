from typing import List



def find_first_even_number(input_list: List[int]) -> int:

    """Finds the index of the first even number in the input list.



    Args:

        input_list: The input list to search for an even number.



    Returns:

        The index of the first even number in the input list, or -1 if no even number is found.

    """

    for i, item in enumerate(input_list):

        if item % 2 == 0:

            return i

    return -1

