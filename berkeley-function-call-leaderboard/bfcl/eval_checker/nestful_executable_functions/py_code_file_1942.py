from typing import AnyStr



def check_beginning_and_end(string: AnyStr) -> bool:

    """

    Compares the characters at the beginning and end of the string.

    Returns a boolean value indicating whether the characters match.



    Args:

        string: The input string.

    """

    length = len(string)

    for i in range(length // 2):

        if string[i] != string[length - i - 1]:

            return False

    return True

