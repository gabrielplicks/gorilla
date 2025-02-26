from typing import Union



def clamp_value(value: Union[int, float], min_value: Union[int, float] = 0, max_value: Union[int, float] = 1) -> Union[int, float]:

    """Clamps a value to a specified range.



    Args:

        value: The value to clamp.

        min_value: The minimum value of the range. Default is 0.

        max_value: The maximum value of the range. Default is 1.



    Returns:

        The clamped value.

    """

    return min(max(value, min_value), max_value)

