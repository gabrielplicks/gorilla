import string

from typing import AnyStr



def is_palindrome_ignore_case(string: AnyStr) -> bool:

    """Checks if a string is a palindrome, ignoring case.



    Args:

        string: The string to check.



    Returns:

        True if the string is a palindrome, False otherwise.

    """

    filtered_string = ''.join(character for character in string if character.isalpha())

    return filtered_string.lower() == filtered_string[::-1].lower()

