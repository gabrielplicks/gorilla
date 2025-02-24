from typing import List



def remove_range(lst: List[int], start: int, end: int) -> List[int]:

    """Removes the elements in the given range from a list.

    Args:

        lst: The list from which to remove the elements.

        start: The start index of the range.

        end: The end index of the range.

    """

    if not isinstance(lst, list):

        raise TypeError("lst must be a list")

    lst_len = len(lst)

    if start < 0:

        start = lst_len + start

    if end < 0:

        end = lst_len + end

    if start < 0 or start >= lst_len or end < 0 or end >= lst_len or start > end:

        raise ValueError("Invalid range")

    return lst[:start] + lst[end+1:]

