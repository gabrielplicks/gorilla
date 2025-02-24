from typing import Optional



def has_one_or_all(string: str, substring1: Optional[str], substring2: Optional[str]) -> bool:

    """Checks if a string contains a certain substring or if it is empty.



    Args:

        string: The string to check.

        substring1: The first substring to check for.

        substring2: The second substring to check for.



    Returns:

        True if the string contains either substring1 or substring2, or if the string is empty and both substrings are also empty. False otherwise.

    """

    if substring1 in string:

        return True

    elif substring2 in string:

        return True

    elif not string and not substring1 and not substring2:

        return True

    else:

        return False

