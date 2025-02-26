from typing import Union



def combine_characters(a: str, b: str) -> str:

    """Combines each character of two strings exactly once, inserting remaining characters at the end.



    Args:

        a: The first string.

        b: The second string.



    Returns:

        A new string that combines each character of the input strings exactly once.

    """

    result = ""

    for i in range(min(len(a), len(b))):

        result += a[i]

        result += b[i]

    if len(a) > len(b):

        result += a[len(b):]

    elif len(b) > len(a):

        result += b[len(a):]



    return result

