from typing import AnyStr



def replace_backslash(string: AnyStr) -> AnyStr:

    """Replaces all backslashes in a string with double backslashes.



    Args:

        string: The input string.



    Returns:

        The output string with all backslashes replaced by double backslashes.

    """

    output = ""

    for char in string:

        if char == "\\":

            output += "\\\\"

        else:

            output += char

    return output

