from typing import List



def find_matching_indices(list1: List[float], list2: List[float]) -> List[int]:

    """Finds the indices in `list1` that are within one unit of any value in `list2`.



    Args:

        list1: A list of numbers.

        list2: A list of numbers.



    Returns:

        A list of indices in `list1` that are within one unit of any value in `list2`.

    """

    matching_indices = []

    for i, value1 in enumerate(list1):

        for value2 in list2:

            if abs(value1 - value2) <= 1:

                matching_indices.append(i)

                break

    return matching_indices

