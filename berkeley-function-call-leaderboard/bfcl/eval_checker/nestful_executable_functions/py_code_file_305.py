from typing import Union



def bitwise_left_shift(value: int, count: Union[int, None] = 1) -> int:

    """

    Performs a Bitwise Left Shift (`<<`) on an integer.

    This function supports two modes of operation:

    - with a single mandatory `value` argument and a single keyword argument `count`,

    - or with two mandatory arguments, `value` and `count`.

    If only a single argument is provided, the function shifts the bits of the integer to the left by one position.

    However, if both arguments are provided, the function shifts the bits by the specified `count` positions.



    Args:

        value: The integer to perform the bitwise left shift on.

        count: The number of positions to shift the bits to the left. Default is 1.

    """

    result = value << count

    return result

