from typing import AnyStr



def is_string_unicode(string: AnyStr) -> bool:

    """Determines if a string is encoded in ASCII or Unicode.



    Args:

        string: The input string to check.



    Returns:

        False if the input string is encoded in ASCII, and True if the string is encoded in Unicode.

    """

    assert string, "Input string must not be empty"

    for char in string:

        code_point = ord(char)

        if code_point > 127:

            return True

    return False

