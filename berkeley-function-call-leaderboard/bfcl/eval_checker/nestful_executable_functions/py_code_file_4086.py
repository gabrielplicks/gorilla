from typing import List



def compare_two_lists(list1: List[str], list2: List[str]) -> List[str]:

    """Returns a list containing items that appear in both list1 and list2 in the same order.



    Args:

        list1: The first list.

        list2: The second list.

    """

    common_elements = []



    for item in list1:

        if item in list2:

            common_elements.append(item)



    return common_elements

