from typing import Union



def get_abs(value: Union[int, bool, str]) -> int:

    """

    Returns the absolute value of the input.

    Args:

        value: The input value.

    Raises:

        TypeError: If the input is not an integer, boolean, or string.

    """

    if isinstance(value, bool):

        return int(value)

    if isinstance(value, int):

        return abs(value)

    if isinstance(value, str):

        try:

            return abs(int(value))

        except ValueError:

            raise TypeError("Input string must be an integer.")

    raise TypeError("Input must be an integer, boolean, or string.")



def get_absolute(x: Union[int, bool, str]) -> int:

    """

    Returns the absolute value of the input.

    Args:

        x: The input value.

    """

    return get_abs(x)

