from typing import AnyStr



def canonicalize_string(str: AnyStr) -> AnyStr:

    """

    Removes leading and trailing whitespaces and parentheses, then removes all whitespaces.

    Args:

        str: The input string.

    """

    str = str.strip().replace("(", "").replace(")", "").replace(" ", "")

    return str

