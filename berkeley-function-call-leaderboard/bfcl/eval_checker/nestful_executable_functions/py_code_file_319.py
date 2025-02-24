from typing import List, Tuple



def sort_by_second_value(lst: List[Tuple[int, int]]) -> List[Tuple[int, int]]:

    """

    Sorts a list of tuples by the second element in ascending order.

    If two or more tuples have the same second element, sort them by the first element in ascending order.

    For example, if the input list is [(1, 3), (2, 1), (3, 2), (4, 3), (5, 1)],

    the output should be [(2, 1), (5, 1), (3, 2), (1, 3), (4, 3)].

    Args:

        lst: A list of tuples, where each tuple contains two integers.

    """

    return sorted(lst, key=lambda x: (x[1], x[0]))

