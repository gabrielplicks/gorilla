from typing import Union



def bits_to_num(s: Union[str, bytes]) -> float:

    """

    Converts a binary string to a decimal number.

    Args:

        s: A string representing a binary number.

    """

    s = s.lstrip('0 ').lstrip('0b')

    return int(s, 2)

