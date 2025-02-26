from typing import List



def str_to_unicode(s: str) -> List[int]:

    """Converts a string to a sequence of integers, where each integer represents a character's Unicode value.

    The conversion supports both ASCII and Unicode characters. For Unicode characters, they are encoded to UTF-8 before calculating their Unicode values.

    Args:

        s: The input string.

    Returns:

        A list of integers representing the Unicode values of the characters in the input string.

    """

    result = []

    for c in s:

        if ord(c) > 127:

            c = c.encode('utf-8')

        result.append(ord(c))

    return result

