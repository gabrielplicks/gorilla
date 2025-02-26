from typing import List



def count_common_elements(list1: List[int], list2: List[int]) -> int:

    """Calculates the number of common elements present in both lists.

    Args:

        list1: The first list of elements.

        list2: The second list of elements.

    """

    set1 = set(list1)

    count = 0

    for item in list2:

        if item in set1:

            count += 1

    return count

