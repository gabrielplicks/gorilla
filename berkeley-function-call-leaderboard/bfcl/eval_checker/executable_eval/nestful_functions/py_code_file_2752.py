from typing import List



def duplicate(x: List) -> List:

    """Duplicates a given list.



    Args:

        x: The input list.



    Raises:

        TypeError: If the input is not a list.



    Returns:

        A new list with each element of the input list duplicated.

    """

    if not isinstance(x, list):

        raise TypeError("Input is not a list!")



    duplicated_list = []

    duplicated_list = [element for element in x for _ in range(2)]

    return duplicated_list

