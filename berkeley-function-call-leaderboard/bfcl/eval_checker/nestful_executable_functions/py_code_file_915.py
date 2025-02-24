from typing import List



def shift_elements(lst: List[int], n: int) -> List[int]:

    """Shifts the first n elements of a list to the end of the list.



    Args:

        lst: The input list.

        n: The number of elements to shift to the end of the list.



    Raises:

        ValueError: If n is larger than the length of the list.

    """

    if len(lst) < n:

        raise ValueError("n is larger than the length of the list")



    return lst[n:] + lst[:n]

