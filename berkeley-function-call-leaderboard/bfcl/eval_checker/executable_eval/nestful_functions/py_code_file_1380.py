from typing import Union



def round_float_to_even(num: Union[float, int]) -> int:

    """

    Rounds a float number to the nearest integer. If the decimal part is exactly 0.5,

    round up to the closest even integer.

    Args:

        num: The float number to be rounded.

    """

    rounded = round(num)



    if rounded % 2 == 0:

        return rounded



    decimal_part = num - rounded

    if decimal_part == 0.5:

        return rounded + 1



    return rounded

