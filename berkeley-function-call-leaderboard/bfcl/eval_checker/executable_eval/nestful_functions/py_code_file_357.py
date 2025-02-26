from typing import Union



def align_left(s: str, length: Union[int, float]) -> str:

    """

    Aligns a string `s` to the left within a string of length `length`.

    If the original string is longer than `length`, it will be truncated from the right.

    If the original string is shorter than `length`, spaces will be added to the right.



    Args:

        s: The string to be aligned.

        length: The desired length of the string. Must be a nonnegative integer.

    """

    assert length >= 0, "Length must be nonnegative"

    return s[:length].ljust(length, ' ')

