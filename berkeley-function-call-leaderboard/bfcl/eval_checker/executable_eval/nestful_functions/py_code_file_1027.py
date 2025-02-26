from typing import AnyStr



def check_characters_order(string: AnyStr) -> AnyStr:

    """Checks if the characters in a string are in alphabetical order.

    If the characters are not in order, the function returns the reversed order sequence.

    If the characters are in order, the function returns the original input string.

    Args:

        string: The input string.

    """

    for i in range(len(string) - 1):

        if string[i] > string[i + 1]:

            return string[::-1]

    return string

