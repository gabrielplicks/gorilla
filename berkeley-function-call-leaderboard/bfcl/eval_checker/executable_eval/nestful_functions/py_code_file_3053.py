from typing import Union



def find_target_string_in_list(lst: Union[list, tuple, str], target: Union[str, int]) -> int:

    """Finds the first index `i` where the string in `lst[i]` matches `target`.



    Args:

        lst: A list, tuple, or string to search.

        target: A string or integer to search for.



    Returns:

        The first index `i` where the string in `lst[i]` matches `target`. If no such index exists, returns -1.

    """

    if isinstance(lst, list) or isinstance(lst, tuple):

        for i, item in enumerate(lst):

            if isinstance(item, str) and item == str(target):

                return i

    elif isinstance(lst, str):

        for i, ch in enumerate(lst):

            if ch == str(target):

                return i

    return -1

