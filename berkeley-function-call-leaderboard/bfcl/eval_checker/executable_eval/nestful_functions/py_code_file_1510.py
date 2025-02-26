from typing import AnyStr



def is_all_uppercase(string: AnyStr) -> bool:

    """Checks if a given string is all uppercase.



    Args:

        string: The string to check.



    Returns:

        A boolean value indicating whether the string is all uppercase.

    """

    for char in string:

        if not char.isupper():

            return False

    return True

