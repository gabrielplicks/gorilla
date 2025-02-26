from typing import List, Union



def max_nested_list_depth(lst: Union[List, int, float, str]) -> int:

    """Calculates the maximum depth of a nested list structure.

    Args:

        lst: The input list.

    Returns:

        The maximum depth of the nested list structure.

    """

    if not isinstance(lst, list):

        return 0

    max_depth = 1  # depth of a non-nested list

    for item in lst:

        max_depth = max(max_depth, 1 + max_nested_list_depth(item))

    return max_depth

