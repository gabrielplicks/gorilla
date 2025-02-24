from typing import List



def contains_all(s: str, l: List[str]) -> bool:

    """Checks whether the string `s` contains all the strings in the list `l`.



    Args:

        s: The string to check.

        l: The list of strings to check for.



    Returns:

        A boolean value indicating whether `s` contains all the strings in `l`.

    """

    chars = set(s)

    for substring in l:

        if not set(substring).issubset(chars):

            return False

    return True

