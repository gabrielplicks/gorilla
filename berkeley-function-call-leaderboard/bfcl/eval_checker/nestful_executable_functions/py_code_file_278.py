from typing import List



def contains_non_empty_strings(strings: List[str]) -> bool:

    """Checks if a list of strings contains only non-empty and non-None strings.



    Args:

        strings: A list of strings to check.



    Returns:

        A boolean indicating whether all strings in the list are non-empty and non-None.

    """

    contains_non_empty_strings = True

    for string in strings:

        if string is None or string == "":

            contains_non_empty_strings = False

            break

    return contains_non_empty_strings

