from typing import List



def parse_unicode(string: str) -> List[int]:

    """Parses a string containing Unicode characters and returns a list of their corresponding bytes.

    The function supports parsing "raw" Unicode strings (prefixed with `u`) or UTF-8 encoded strings (prefixed with `b`).

    It handles the conversion of Unicode characters with no direct UTF-8 encoding.

    List comprehension is used to construct the list of bytes from the Unicode string.



    Args:

        string: The input string containing Unicode characters.



    Returns:

        A list of bytes representing the Unicode characters in the input string.

    """

    return [b for b in string.encode('utf-8', 'replace')]

