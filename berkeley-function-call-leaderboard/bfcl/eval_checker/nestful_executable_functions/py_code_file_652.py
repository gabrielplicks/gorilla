from typing import List



def transpose_nested_list(nested_list: List[List[int]]) -> List[List[int]]:

    """Transposes a nested list of equal-length lists.



    Args:

        nested_list: A nested list of equal-length lists.



    Returns:

        The transposed nested list.

    """

    if not isinstance(nested_list, list) or not all(isinstance(sublist, list) for sublist in nested_list):

        return []

    if not all(len(sublist) == len(nested_list[0]) for sublist in nested_list):

        return []

    return [[nested_list[j][i] for j in range(len(nested_list))] for i in range(len(nested_list[0]))]

