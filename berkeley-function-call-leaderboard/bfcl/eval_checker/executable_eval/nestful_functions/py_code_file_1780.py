from typing import List



def add_suffix_to_strings(strings: List[str], suffix: str) -> List[str]:

    """Adds a suffix to each string in a list of strings and returns a new list.



    Args:

        strings: A list of strings to which the suffix will be added.

        suffix: The suffix to be added to each string.



    Returns:

        A new list of strings with the suffix added to each string.

    """

    strings_with_suffix = strings[:]

    for i, s in enumerate(strings):

        strings_with_suffix[i] = s + suffix

    return strings_with_suffix

