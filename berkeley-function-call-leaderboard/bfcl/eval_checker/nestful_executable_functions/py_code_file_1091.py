from typing import List



def get_sum(my_list: List[int]) -> int:

    """Calculates the sum of the first and last elements of a list of integers.

    If the list is empty, returns 0. If the list contains only one element, returns it.

    Args:

        my_list: A list of integers.

    """

    if len(my_list) == 0:

        return 0

    elif len(my_list) == 1:

        return my_list[0]

    else:

        return my_list[0] + my_list[-1]

