from typing import List, Tuple



def find_min_max_indices(lst: List[int]) -> Tuple[int, int]:

    """Finds the indices of the minimum and maximum values in the given list.

    Args:

        lst: The input list.

    Returns:

        A tuple containing the indices of the minimum and maximum values.

    """

    if not lst:

        return ()

    min_idx = 0

    max_idx = 0



    for i, x in enumerate(lst):

        if x < lst[min_idx]:

            min_idx = i

        if x > lst[max_idx]:

            max_idx = i



    return (min_idx, max_idx)

