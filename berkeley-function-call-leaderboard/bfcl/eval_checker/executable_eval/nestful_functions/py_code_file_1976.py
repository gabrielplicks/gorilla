from typing import AnyStr



def check_special_characters(string: AnyStr) -> bool:

    """Checks if a string contains any of the following characters: `@` or `#` or `$` or `%` or `^`.



    Args:

        string: The string to check.



    Returns:

        True if the string does not contain any of the special characters, False otherwise.

    """

    return not any(char in '@#$%^' for char in string)

