from typing import AnyStr



def pad_truncate_with_char(s: AnyStr, char: str, max_length: int) -> AnyStr:

    """Pads or truncates a string with a specified character until it reaches a maximum length.



    Args:

        s: The input string.

        char: The character to pad the string with.

        max_length: The maximum length of the returned string.



    Returns:

        The modified string.

    """

    if len(s) < max_length:

        return s + (char * (max_length - len(s)))

    else:

        return s[:max_length]

