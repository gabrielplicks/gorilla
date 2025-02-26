from typing import List, Tuple



def pair_vars_with_rotated_list(lst: List[int]) -> List[Tuple[int, int]]:

    """Creates a list of tuples, where each tuple contains a variable and its rotated version.

    Args:

        lst: A list of variables.

    Returns:

        A list of tuples.

    """

    if not lst or len(lst) == 1:

        return []

    result = []

    for idx, var in enumerate(lst):

        if idx == len(lst) - 1:

            rotated_var = lst[0]

        else:

            rotated_var = lst[idx + 1]

        result.append((var, rotated_var))

    return result

