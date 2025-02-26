from typing import AnyStr



def remove_non_digits(s: AnyStr) -> str:

    """Removes all characters from a given string that are not numbers (0-9).

    Args:

        s: The input string.

    """

    result = ""

    for c in s:

        if c.isdigit():

            result += c

    return result

