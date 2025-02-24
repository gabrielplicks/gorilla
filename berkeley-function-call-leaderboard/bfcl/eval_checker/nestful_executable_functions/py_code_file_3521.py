from typing import List



def list_in_range(list_of_strings: List[str], start: int, end: int) -> List[str]:

    """Returns a new list of strings that only contain the strings within the given range, from `start` to `end` (inclusive).



    Args:

        list_of_strings: The input list of strings.

        start: The start index of the range.

        end: The end index of the range.

    """

    new_list = []

    for i, string in enumerate(list_of_strings):

        if start <= i <= end:

            new_list.append(string)

    return new_list

