from typing import Union



def repeat_str(n: Union[int, None], string: str) -> str:

    """

    Returns a new string with n copies of the original string.

    Args:

        n: The number of times to repeat the string.

        string: The string to repeat.

    """

    if not isinstance(n, int) or n < 1:

        return ""

    return string * n

