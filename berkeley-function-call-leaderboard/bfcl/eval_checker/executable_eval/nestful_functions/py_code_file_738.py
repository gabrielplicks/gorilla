from typing import Union



def pad_to_len(num: Union[int, float]) -> str:

    """Pads a number with leading zeroes so that it has at least 4 digits in length.



    Args:

        num: The number to pad.



    Returns:

        The padded number as a string.

    """

    return str(num).zfill(4)



def f(x: Union[int, float], y: Union[int, float]) -> int:

    """Calculates the value of the function f(x, y) using the given formula.



    Args:

        x: The first input to the function.

        y: The second input to the function.



    Returns:

        The value of the function f(x, y).

    """

    return int(pad_to_len(x) + str(y)) % 1000

