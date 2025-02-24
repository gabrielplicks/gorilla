from typing import Union



def last_two_characters(string: str) -> str:

    """Returns a new string composed of the last 2 characters of the input string.

    If the input string is shorter than 2 characters, it returns the string with characters in reverse order.

    Args:

        string: The input string.

    """

    if len(string) < 2:

        return string[::-1]

    else:

        return string[-2:]

