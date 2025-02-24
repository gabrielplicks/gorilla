from typing import Optional



def trim_trailing_dot(string: str) -> Optional[str]:

    """Trims the trailing period (dot) character from a given string.



    Args:

        string: The input string.



    Returns:

        The input string without the trailing period (dot) character, or the original string if it does not end with a period.

    """

    if string.endswith('.'):

        return string[:-1]

    return string

