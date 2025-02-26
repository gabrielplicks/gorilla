from typing import AnyStr



def validate_ascii(string: AnyStr) -> bool:

    """Validates if all characters in a string are ASCII characters.



    Args:

        string: The input string to validate.



    Returns:

        True if all characters are ASCII characters, otherwise False.

    """

    for char in string:

        if ord(char) > 127:

            return False

    return True

