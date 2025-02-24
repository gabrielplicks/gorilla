import itertools

from typing import List



def merge_and_sort_lists(list1: List[int], list2: List[int]) -> List[int]:

    """Merges two lists of numbers and sorts them in ascending order.



    Args:

        list1: A list of numbers.

        list2: A list of numbers.



    Returns:

        A new list containing all the numbers from both input lists, sorted in ascending order.

    """

    merged_list = sorted(itertools.chain(list1, list2))

    return merged_list

