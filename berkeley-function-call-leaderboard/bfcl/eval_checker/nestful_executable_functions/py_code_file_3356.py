from typing import List



def remove_first_n(lst: List, n: int) -> List:

    """Removes the first `n` elements from a list and returns a new list.



    Args:

        lst: The input list.

        n: The number of elements to remove from the beginning of the list.



    Raises:

        TypeError: If the input list is not a list.

        ValueError: If `n` is not a positive integer.

    """

    if not isinstance(lst, list):

        raise TypeError('Input list must be a list')

    if n < 0:

        raise ValueError('n must be a positive integer')

    if len(lst) == 0:

        return []



    return list(lst[n:])

