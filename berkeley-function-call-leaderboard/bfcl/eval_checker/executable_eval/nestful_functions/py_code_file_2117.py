from typing import List



def get_strings_containing_char(my_strings: List[str], my_char: str) -> List[str]:

    """

    Returns a new list with only the strings that contain a specific character.

    Args:

        my_strings: A list of strings.

        my_char: The character to search for.

    """

    filtered_strings = []

    for string in my_strings:

        if my_char in string:

            filtered_strings.append(string)

    return filtered_strings

