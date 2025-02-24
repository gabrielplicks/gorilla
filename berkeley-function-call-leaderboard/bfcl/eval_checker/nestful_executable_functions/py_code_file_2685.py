import re

from typing import Union



def wrap_with_quotes(string: str) -> Union[str, str]:

    """Wraps a string with double quotes, and if the string contains a double quote character, wraps it with single quotes.



    Args:

        string: The string to be wrapped.



    Returns:

        The wrapped string.

    """

    if re.search(r'"', string):

        return f"'{string}'"

    return f'"{string}"'

