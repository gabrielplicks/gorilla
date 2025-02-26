from typing import List, Union



def starts_with_one_of(string: str, prefixes: List[str]) -> bool:

    """Checks if a string starts with one of the specified prefixes.



    Args:

        string: The string to check.

        prefixes: A list of prefixes to check against.

    """

    for prefix in prefixes:

        if string.startswith(prefix):

            return True

    return False

