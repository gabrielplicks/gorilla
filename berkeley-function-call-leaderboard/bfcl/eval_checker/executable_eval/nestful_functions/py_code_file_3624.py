from typing import AnyStr



def contains_only_lowercase(string: AnyStr) -> bool:

    """Checks if a string contains only lowercase letters.



    Args:

        string: The input string.



    Returns:

        True if the string contains only lowercase letters, False otherwise.

    """

    for char in string:

        if not char.islower():

            return False

    return True

