from typing import List



def split_into_parts(lst: List[int]) -> tuple:

    """Splits a list into two parts with equal length, but the first part may be longer than the second part by one element.

    The function maintains the order of the elements in the original list.

    Args:

        lst: The list to be split.

    Returns:

        A tuple containing the two parts of the list.

    """

    mid = len(lst) // 2

    first_part = lst[:mid + 1]

    second_part = lst[mid + 1:]

    return first_part, second_part

