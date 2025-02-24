import re



def is_valid_variable_length(string: str) -> bool:

    """Checks whether the given string is a valid variable length.



    A variable length is considered valid if it follows the following rules:

    1. It should be a string of 1 to 32 characters.

    2. It should not match the regular expression `^.*\d+.*$`.

    3. It should contain only uppercase and lowercase letters, digits, and underscores.



    Args:

        string: The string to check.



    Returns:

        A boolean value indicating whether the string is a valid variable length.

    """

    if len(string) < 1 or len(string) > 32:

        return False

    if any(char.isdigit() for char in string):

        return False

    if any(not char.isalnum() and char != '_' for char in string):

        return False

    return True

