from typing import List



def identical_fraction(list1: List[object], list2: List[object]) -> float:

    """Calculates the fraction of identical elements between two lists.



    Args:

        list1: The first list of elements.

        list2: The second list of elements.



    Returns:

        The fraction of identical elements between the two lists.

    """

    if not list1 or not list2:

        return 0



    unique_elements = set(list1 + list2)

    common_elements = len(set(list1) & set(list2))



    return common_elements / len(unique_elements)

