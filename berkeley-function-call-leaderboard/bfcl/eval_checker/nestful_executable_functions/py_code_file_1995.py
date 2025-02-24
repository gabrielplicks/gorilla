from typing import AnyStr



def remove_non_alpha(string: AnyStr) -> AnyStr:

    """Removes all non-alphabetic characters from a string.



    Args:

        string: The input string.



    Returns:

        The input string with all non-alphabetic characters removed.

    """

    new_string = ""

    for char in string:

        if char.isalpha():

            new_string += char

    return new_string

