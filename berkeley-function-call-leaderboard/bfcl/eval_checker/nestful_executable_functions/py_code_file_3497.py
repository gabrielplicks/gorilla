from typing import Union



def ceiling_integer(num: Union[int, float]) -> int:

    """Computes the ceiling of the integer part of the given non-negative floating-point number.



    Args:

        num: The non-negative floating-point number.



    Returns:

        The ceiling integer of the input number.

    """

    if num == int(num):

        return num

    else:

        return int(num) + 1

