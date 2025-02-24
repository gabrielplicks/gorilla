from typing import List



def get_zero_based_index(lst: List[int], one_based_index: int) -> int:

    """Converts a one-indexed integer to a zero-indexed integer for a given list.



    Args:

        lst: The input list.

        one_based_index: The one-indexed integer to be converted.



    Returns:

        The zero-indexed integer corresponding to the one-indexed integer in the list.

    """

    return one_based_index - 1

