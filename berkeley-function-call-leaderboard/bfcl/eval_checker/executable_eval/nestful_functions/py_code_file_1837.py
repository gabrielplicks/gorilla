from typing import List



def add_n_to_list(lst: List[int], n: int) -> List[int]:

    """Adds a given integer to a list of integers and returns the resulting list.

    If the input list is empty, the function throws a ValueError exception.

    Args:

        lst: The input list of integers.

        n: The integer to be added to each element in the list.

    """

    if not lst:

        raise ValueError("Input list is empty")

    return [item + n for item in lst]

