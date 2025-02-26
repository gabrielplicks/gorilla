from typing import List



def check_at_least_one_element_exists(list1: List[int], list2: List[int]) -> bool:

    """Checks if at least one element in the first list exists in the second list.



    Args:

        list1: The first list.

        list2: The second list.



    Returns:

        True if at least one element in the first list exists in the second list, False otherwise.

    """

    set2 = set(list2)

    if not list1:

        return False

    if not list2:

        return True

    for element in list1:

        if element in set2:

            return True

    return False

