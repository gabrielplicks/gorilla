from typing import Union



def append_string(string: Union[str, None], append_str: str, default_string: str = '') -> str:

    """Appends a string to the end of the original string.



    Args:

        string: The original string.

        append_str: The string to be appended.

        default_string: The default string to return if the original string is not a string.



    Returns:

        The original string with the appended string.

    """

    if not isinstance(string, str) or not string:

        return default_string

    if not append_str:

        return string

    return string + append_str

