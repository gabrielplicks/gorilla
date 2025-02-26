from typing import Union



def filter_non_ascii_characters(unicode_string: Union[str, bytes]) -> str:

    """Filters out non-ASCII characters from a Unicode string and returns a new string containing only ASCII characters.



    Args:

        unicode_string: The Unicode string to filter.



    Returns:

        A new string containing only ASCII characters from the given Unicode string.

    """

    ascii_string = ""

    for character in unicode_string:

        if ord(character) < 128:

            ascii_string += character

    return ascii_string

