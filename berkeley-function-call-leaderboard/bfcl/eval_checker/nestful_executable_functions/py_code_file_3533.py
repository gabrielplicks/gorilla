from typing import AnyStr



def escape_non_alphanumeric(s: AnyStr) -> AnyStr:

    """Escapes all non-alphanumeric characters in a string.



    Args:

        s: The input string.



    Returns:

        A modified string where all non-alphanumeric characters are escaped.

    """

    modified_string = ""

    for c in s:

        if c.isalnum():

            modified_string += c

        else:

            modified_string += f"\\x{hex(ord(c))[2:]}"

    return modified_string

