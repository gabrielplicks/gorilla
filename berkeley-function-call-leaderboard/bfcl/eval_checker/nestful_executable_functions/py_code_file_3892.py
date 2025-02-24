from typing import Union



def round_to_int(float_value: Union[float, int]) -> int:

    """Rounds a float to the nearest integer.

    If the float is negative, then the function returns the largest integer less than or equal to the float.

    Otherwise, the function returns the smallest integer greater than or equal to the float.

    Args:

        float_value: The float to be rounded.

    """

    if float_value < 0:

        return int(float_value - 0.5)

    else:

        return int(float_value + 0.5)

