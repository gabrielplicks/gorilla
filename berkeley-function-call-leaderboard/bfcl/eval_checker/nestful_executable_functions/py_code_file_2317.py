from typing import List



def select_longest_string(strings: List[str]) -> str:

    """Selects the longest string from a list of strings.



    Args:

        strings: A list of strings.



    Returns:

        The longest string from the list.

    """

    sorted_strings = sorted(strings, key=lambda x: len(x), reverse=True)

    return sorted_strings[0]

