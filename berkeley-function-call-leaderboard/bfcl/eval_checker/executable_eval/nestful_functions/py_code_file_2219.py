from typing import List



def filter_list_without_loops(num_list: List[int]) -> List[int]:

    """Filters a list of integers to remove even numbers without using loops.



    Args:

        num_list: The list of integers to filter.



    Returns:

        The filtered list of integers.

    """

    return [num for num in num_list if num % 2 != 0]

