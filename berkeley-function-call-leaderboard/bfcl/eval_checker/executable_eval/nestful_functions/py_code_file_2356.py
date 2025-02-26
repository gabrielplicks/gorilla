from typing import AnyStr



def has_consecutive_duplicates(s: AnyStr) -> bool:

    """Determines whether a string `s` contains any consecutive duplicate characters.



    Args:

        s: The input string.



    Returns:

        A boolean value indicating whether `s` contains two or more consecutive duplicate characters.

    """

    for i in range(len(s) - 1):

        if s[i] == s[i+1]:

            return True

    return False

