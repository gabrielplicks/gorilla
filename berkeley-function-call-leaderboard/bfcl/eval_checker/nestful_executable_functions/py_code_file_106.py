from typing import AnyStr



def convert_to_opposite_case(text: AnyStr) -> AnyStr:

    """Converts a string to the opposite case.



    Args:

        text: The input string.



    Returns:

        The input string with all lower-case characters replaced by their upper-case counterparts and vice-versa.

    """

    result = ""

    for char in text:

        if ord(char) >= 65 and ord(char) <= 90:

            result += chr(ord(char) + 32)

        elif ord(char) >= 97 and ord(char) <= 122:

            result += chr(ord(char) - 32)

        else:

            result += char

    return result

