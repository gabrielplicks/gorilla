from typing import Dict, Set



def assign_indices(lst: List[int]) -> Dict[int, int]:

    """Determines the unique elements in a list and assigns them to integer indices starting from 0.

    Returns a dictionary where the keys are the unique elements and the values are the assigned indices.

    Args:

        lst: The input list.

    """

    unique_elements: Set[int] = set(lst)

    result: Dict[int, int] = {}

    for index, element in enumerate(unique_elements):

        result[element] = index

    return result

