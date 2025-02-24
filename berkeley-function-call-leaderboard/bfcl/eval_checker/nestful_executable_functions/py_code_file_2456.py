from typing import List



def find_first_string_starting_with_A(strings: List[str]) -> str:

    """Finds the first string that starts with an uppercase 'A' in a given list of strings.



    Args:

        strings: A list of strings.



    Returns:

        The first string that starts with an uppercase 'A', or an empty string if no such string exists.

    """

    filtered_strings = [string for string in strings if string.startswith('A')]

    if not filtered_strings:

        return ''

    return filtered_strings[0]

