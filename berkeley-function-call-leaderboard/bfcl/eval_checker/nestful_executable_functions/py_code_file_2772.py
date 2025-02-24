from typing import List



def unique_intersection(list1: List[int], list2: List[int]) -> List[int]:

    """Generates a list of unique numbers from two lists by taking the intersection of them.



    Args:

        list1: The first list of numbers.

        list2: The second list of numbers.



    Returns:

        A new list with unique numbers in sorted order.

    """

    set1 = set(list1)

    set2 = set(list2)

    return sorted(set1.intersection(set2))

