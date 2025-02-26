from typing import AnyStr



def count_bytes(string: AnyStr) -> int:

    """Calculates the number of bytes needed to encode a string as ASCII.



    Args:

        string: The input string.



    Returns:

        The number of bytes needed to encode the string as ASCII.

    """

    num_bytes = 0

    for char in string:

        num_bytes += 1

    return num_bytes

