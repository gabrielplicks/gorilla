from typing import AnyStr



def remove_adjacent_duplicates(string: AnyStr) -> AnyStr:

    """Removes all adjacent duplicate characters from a given string.



    Args:

        string: The input string.



    Returns:

        The altered string with adjacent duplicates removed.

    """

    new_string = ""

    last_seen = None

    for char in string:

        if char != last_seen:

            new_string += char

        last_seen = char

    return new_string

