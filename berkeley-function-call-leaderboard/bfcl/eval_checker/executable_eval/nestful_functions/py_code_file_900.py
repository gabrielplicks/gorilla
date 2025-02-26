from typing import Union



def pad_string_with_leading_zeros(s: str, length: int) -> str:

    """Pads a string with leading zeros to a specific length.



    Args:

        s: The input string.

        length: The desired length of the output string.



    Returns:

        A new string with `length` characters, where the input string is right aligned and padded with zeros if necessary.

    """

    return s.zfill(length)

