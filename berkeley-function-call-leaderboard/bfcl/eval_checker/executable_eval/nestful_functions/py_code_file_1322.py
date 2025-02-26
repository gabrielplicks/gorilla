from typing import Union



def color_value(value: Union[int, float], default: str) -> str:

    """

    Returns a color for the given value.

    If the value is positive, the function returns the string 'red'.

    If the value is negative, the function returns the string 'green'.

    If the value is equal to 0, the function returns the string 'blue'.

    Otherwise, the function returns the default value.



    Args:

        value: The value to determine the color for.

        default: The default color to return if the value does not match any of the conditions.

    """

    color_map = {

        0: 'blue',

        -1: 'green',

        1: 'red'

    }

    return color_map.get(value, default)

