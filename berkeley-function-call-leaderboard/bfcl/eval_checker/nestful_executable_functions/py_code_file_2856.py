from typing import List



def list_length(data: List[Any]) -> int:

    """Calculates the length of a list without using the built-in `len()` function.



    Args:

        data: The list to calculate the length of.



    Returns:

        The length of the list.

    """

    length = 0

    for element in data:

        length += 1

    return length

