from typing import List



def unique_and_sorted(lst: List[int]) -> List[int]:

    """Creates a new list that contains all unique elements of the given list, sorted in ascending order.



    Args:

        lst: A list of integers.



    Returns:

        A new list that contains all unique elements of the given list, sorted in ascending order.

    """

    unique_lst = list(set(lst))

    unique_lst.sort()

    return unique_lst

