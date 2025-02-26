from typing import Union



def make_python_str(s: Union[str, bytes]) -> str:

    """

    Converts a string or bytes object to a valid Python string.



    Args:

        s: The input string or bytes object.



    Returns:

        A valid Python string.

    """

    result = '"'

    for char in s:

        if char == '"':

            result += '\\"'

        elif char == '\\':

            result += '\\\\'

        elif char == '\n':

            result += '\\n'

        else:

            result += char

    result += '"'

    return result

