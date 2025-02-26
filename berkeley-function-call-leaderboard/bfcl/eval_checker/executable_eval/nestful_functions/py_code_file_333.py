from typing import List



def reverse_string_list(string_list: List[str]) -> List[str]:

    """Reverses the order of a list of strings.



    Args:

        string_list: A list of strings.



    Returns:

        A reversed copy of the input list. If the input list is empty, returns an empty list.

        If the input list contains only the string "N/A", returns a list with only the string "N/A".

    """

    if not string_list:

        return []

    elif len(string_list) == 1 and string_list[0] == "N/A":

        return ["N/A"]

    else:

        return string_list[::-1]

