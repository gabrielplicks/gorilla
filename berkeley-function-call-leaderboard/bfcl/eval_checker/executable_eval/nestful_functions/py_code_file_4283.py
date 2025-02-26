from typing import AnyStr



def pad_string_left(s: AnyStr, n: int) -> AnyStr:

    """Pads a string on the left with spaces until it becomes `n` characters long.



    Args:

        s: The string to pad.

        n: The desired length of the padded string.



    Returns:

        The padded string.

    """

    if len(s) < n:

        return s.rjust(n, ' ')  # Pad on the left with spaces

    return s  # If length of s is already greater than or equal to n, return s as is

