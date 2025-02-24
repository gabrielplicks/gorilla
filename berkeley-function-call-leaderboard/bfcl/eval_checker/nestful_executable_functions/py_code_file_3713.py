from typing import Union



def trim_list(s: str, n: Union[int, str]) -> Union[str, None]:

    """Trims a string to a specified length.



    Args:

        s: The string to trim.

        n: The length to trim the string to. If n is not a non-negative integer,

           or if n is larger than the length of s, the entire string is returned.



    Returns:

        The trimmed string, or None if n is not a non-negative integer.

    """

    if not isinstance(n, int) or n < 0:

        return None

    if n > len(s):

        return s

    return s[:n]

