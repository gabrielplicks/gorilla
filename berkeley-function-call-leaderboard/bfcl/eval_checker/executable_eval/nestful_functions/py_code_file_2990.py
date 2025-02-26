from typing import List, Union



def sort_list_descending(numbers: List[Union[int, float, str]]) -> List[Union[int, float, str]]:

    """Sorts a list in descending order.

    If the first element of the list is a string, it sorts the list alphabetically.

    Otherwise, it sorts the list numerically.

    If the list is empty, it returns an empty list.



    Args:

        numbers: A list of numbers or strings.



    Returns:

        A list sorted in descending order.

    """

    if not numbers:

        return []



    if isinstance(numbers[0], str):

        return sorted(numbers, reverse=True)

    else:

        return sorted(numbers, reverse=True)

