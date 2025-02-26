from typing import List



def min_indices(a: List[int]) -> List[int]:

    """Returns a list of indices for which the corresponding value is the minimum value in `a`.



    Args:

        a: A non-empty list of integers.



    Returns:

        A list of indices.

    """

    min_value = min(a)

    min_indices = []

    for i, value in enumerate(a):

        if value == min_value:

            min_indices.append(i)

    return min_indices

