from typing import AnyStr



def convert_camel_to_underscore(text: AnyStr) -> AnyStr:

    """Converts a string in camel case to the corresponding string in snake case.



    Args:

        text: The input string in camel case.



    Returns:

        The input string converted to snake case.

    """

    result = []

    for char in text:

        if char.isupper():

            result.append('_')

            result.append(char.lower())

        else:

            result.append(char)

    return ''.join(result)

