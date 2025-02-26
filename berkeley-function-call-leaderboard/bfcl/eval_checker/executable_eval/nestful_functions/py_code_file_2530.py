from typing import List



def swap_by_index(values: List[int], i: int, j: int) -> List[int]:

    """Swaps the values at index i and j in the given list of integers.



    Args:

        values: The list of integers.

        i: The index of the first value to swap.

        j: The index of the second value to swap.

    """

    values[i], values[j] = values[j], values[i]

    return values

