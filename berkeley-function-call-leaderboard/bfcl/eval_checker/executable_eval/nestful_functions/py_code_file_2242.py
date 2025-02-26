from typing import AnyStr



def first_part(string: AnyStr, separator: AnyStr) -> AnyStr:

    """Returns the first part of a string up to a given separator.



    Args:

        string: The input string.

        separator: The separator character.



    Returns:

        The first part of the string before the separator.

    """

    parts = string.split(separator)

    return parts[0]

