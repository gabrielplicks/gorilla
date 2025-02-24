from typing import AnyStr



def opposite_case(text: AnyStr) -> AnyStr:

    """Converts a given string to a new string where each lowercase character is replaced with the uppercase version of that character and each uppercase character is replaced with the lowercase version of that character.



    Args:

        text: The input string.



    Returns:

        The converted string.

    """

    result = ''

    for char in text:

        if char.islower():

            result += char.upper()

        elif char.isupper():

            result += char.lower()

        else:

            result += char

    return result

