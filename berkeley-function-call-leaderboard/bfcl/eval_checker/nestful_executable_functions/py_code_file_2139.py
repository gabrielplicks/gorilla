from typing import List



def concatenate_and_count(list_: List[str]) -> str:

    """

    Concatenates the elements of the list into a single string, separated by spaces,

    and appends the number of elements in the list enclosed in square brackets at the end.



    Args:

        list_: The list of strings to concatenate.



    Returns:

        A single string containing all the elements of the list, separated by spaces,

        with the number of elements in the list enclosed in square brackets at the end.

    """

    if not list_:

        return ''



    return ' '.join(list_) + ' [' + str(len(list_)) + ']'

