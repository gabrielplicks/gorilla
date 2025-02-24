from typing import AnyStr



def filter_punctuation(s: AnyStr) -> AnyStr:

    """Filters out punctuation from a string.



    Args:

        s: The input string.



    Returns:

        The input string without punctuation.

    """

    filtered = ''

    for c in s:

        if c.isalpha() or c.isspace():

            filtered += c

    return filtered

