from typing import AnyStr



def escape_single_quotes_and_backslash(string: AnyStr) -> AnyStr:

    """Escapes single quotes and backslashes in a string.



    Args:

        string: The input string.



    Returns:

        A new string with all single quotes and backslashes escaped.

    """

    string = string.replace("'", "''")  # Replace single quotes with two single quotes

    string = string.replace("\\", "\\\\")  # Replace backslashes with four backslashes

    return string

