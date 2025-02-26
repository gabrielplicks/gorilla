from typing import Union



def to_ascii_digit(val: Union[int, float]) -> str:

    """Converts a numeric value into the corresponding ASCII digit character.



    Args:

        val: The numeric value to convert.



    Returns:

        The corresponding ASCII character for the given number.

    """

    return chr(ord('0') + int(val))

