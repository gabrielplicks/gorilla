from typing import Union



def to_percentage(value: Union[float, int]) -> str:

    """Converts a value of type float or int to a percentage string, rounding to the nearest integer.

    Args:

        value: The value to convert to a percentage string.

    """

    if not isinstance(value, (float, int)):

        raise TypeError("Input must be of type float or int")

    return f"{round(value * 100)}%"

